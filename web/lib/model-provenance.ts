import fs from 'fs';
import crypto from 'crypto';
import { db } from '@/lib/db';

export type Platform = 'esp32' | 'wyoming';

export interface RunInfo {
  id: number;
  status: string;
  createdAt: string;
  finishedAt: string | null;
}

export interface Provenance {
  /** Run whose completion time matches the file (the run that produced it), if any. */
  producedBy: RunInfo | null;
  /** Newest run for this model + platform, regardless of status. */
  latestRun: RunInfo | null;
  /** True when the file on disk can NOT be the newest training result. */
  stale: boolean;
  staleReason: 'training_running' | 'newer_run_no_file' | 'latest_run_failed' | null;
}

/** Tolerance between run finish and file mtime (tflite/onnx is written in the last seconds of a run). */
const MATCH_WINDOW_MS = 15 * 60 * 1000;

export function fileMd5(fp: string): string {
  return crypto.createHash('md5').update(fs.readFileSync(fp)).digest('hex');
}

function toRunInfo(r: { id: number; status: string; createdAt: Date; finishedAt: Date | null }): RunInfo {
  return { id: r.id, status: r.status, createdAt: r.createdAt.toISOString(), finishedAt: r.finishedAt?.toISOString() ?? null };
}

function platformFilter(platform: Platform): string[] {
  return platform === 'esp32' ? ['microWakeWord', 'both'] : ['openWakeWord', 'both'];
}

/**
 * Cross-check a model file against the training-run database.
 * Guarantees the GUI never presents an old file as "current" when a newer run exists.
 */
export async function getProvenance(modelName: string, platform: Platform, fileMtime: Date): Promise<Provenance> {
  const runs = await db.trainingRun.findMany({
    where: { modelName, platform: { in: platformFilter(platform) } },
    orderBy: { id: 'desc' },
    take: 50,
  });

  const latest = runs[0] ?? null;
  const mtime = fileMtime.getTime();

  const produced = runs.find(r =>
    r.status === 'done' && r.finishedAt &&
    Math.abs(r.finishedAt.getTime() - mtime) <= MATCH_WINDOW_MS,
  ) ?? null;

  let stale = false;
  let staleReason: Provenance['staleReason'] = null;
  if (latest) {
    if (latest.status === 'running' || latest.status === 'paused') {
      stale = true; staleReason = 'training_running';
    } else if (latest.status === 'done' && latest.finishedAt && latest.finishedAt.getTime() - mtime > MATCH_WINDOW_MS) {
      stale = true; staleReason = 'newer_run_no_file';
    } else if (latest.status === 'failed' && latest.createdAt.getTime() > mtime && (!produced || latest.id > produced.id)) {
      stale = true; staleReason = 'latest_run_failed';
    }
  }

  return {
    producedBy: produced ? toRunInfo(produced) : null,
    latestRun: latest ? toRunInfo(latest) : null,
    stale,
    staleReason,
  };
}
