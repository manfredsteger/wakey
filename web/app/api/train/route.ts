import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { db } from '@/lib/db';
import { LOGS_DIR, OUTPUT_DIR, PYTHON, TRAIN_SCRIPT } from '@/lib/paths';
import { readSettings } from '@/lib/settings';

function isPidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

/** Recover runs whose process died while the server was down. */
async function recoverOrphanedRuns() {
  const stale = await db.trainingRun.findMany({
    where: { status: { in: ['running', 'paused'] } },
  });
  for (const run of stale) {
    if (run.pid && !isPidAlive(run.pid)) {
      let logContent = '';
      try { logContent = run.logFile ? fs.readFileSync(run.logFile, 'utf8') : ''; } catch { /* ignore */ }
      const finalStatus = logContent.includes('__DONE__') ? 'done' : 'failed';
      await db.trainingRun.update({
        where: { id: run.id },
        data: { status: finalStatus, finishedAt: new Date() },
      }).catch(() => {});
      if (run.logFile) {
        try { fs.appendFileSync(run.logFile, `\n__${finalStatus.toUpperCase()}__\n`); } catch { /* ignore */ }
      }
    }
  }
}

let _recovered = false;

export async function GET() {
  // Run once per server lifecycle to recover runs left stuck by a server restart
  if (!_recovered) {
    _recovered = true;
    recoverOrphanedRuns().catch(() => {});
  }
  const runs = await db.trainingRun.findMany({
    orderBy: { createdAt: 'desc' },
    take: 20,
  });
  return NextResponse.json(runs);
}

export async function POST(req: Request) {
  const { wakeWord, samples, steps, full, platform } = await req.json() as {
    wakeWord: string;
    samples: number;
    steps: number;
    full: boolean;
    platform: string;
  };

  if (!wakeWord?.trim()) return NextResponse.json({ error: 'wakeWord required' }, { status: 400 });

  // Block concurrent training runs
  const existing = await db.trainingRun.findFirst({ where: { status: 'running' } });
  if (existing) {
    return NextResponse.json({ error: 'training_already_running', id: existing.id }, { status: 409 });
  }

  const modelName = wakeWord.toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');

  // Detect real voice recordings already present for this wake word
  const realVoiceDir = path.join(OUTPUT_DIR, modelName, 'positive_train');
  let hasRealVoice = false;
  try {
    hasRealVoice = fs.readdirSync(realVoiceDir).some(f => f.startsWith('real_'));
  } catch { /* dir doesn't exist yet */ }

  const resolvedPlatform = platform === 'microWakeWord' ? 'microWakeWord' : platform === 'both' ? 'both' : 'openWakeWord';

  const run = await db.trainingRun.create({
    data: { wakeWord, modelName, samples, steps, fullMode: full, status: 'running', hasRealVoice, platform: resolvedPlatform },
  });

  fs.mkdirSync(LOGS_DIR, { recursive: true });
  const logFile = path.join(LOGS_DIR, `run_${run.id}.log`);
  fs.writeFileSync(logFile, '');

  await db.trainingRun.update({ where: { id: run.id }, data: { logFile } });

  const { ttsLang } = readSettings();
  const args = [TRAIN_SCRIPT, wakeWord, '--samples', String(samples), '--steps', String(steps), '--platform', resolvedPlatform, '--lang', ttsLang ?? 'de'];
  if (full) args.push('--full');

  const logFd = fs.openSync(logFile, 'a');
  const child = spawn(PYTHON, args, {
    detached: true,
    stdio: ['ignore', logFd, logFd],
    env: { ...process.env, PYTHONUNBUFFERED: '1', FORCE_COLOR: '0' },
  });
  child.unref();

  await db.trainingRun.update({ where: { id: run.id }, data: { pid: child.pid } });

  child.on('close', async (code) => {
    try { fs.closeSync(logFd); } catch { /* already closed */ }
    const status = code === 0 ? 'done' : 'failed';
    try {
      await db.trainingRun.update({
        where: { id: run.id },
        data: { status, finishedAt: new Date() },
      });
    } catch { /* dev server may have hot-reloaded */ }
    try {
      fs.appendFileSync(logFile, `\n__${status.toUpperCase()}__\n`);
    } catch { /* log may be gone */ }
  });

  return NextResponse.json({ id: run.id });
}
