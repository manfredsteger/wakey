import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { db } from '@/lib/db';
import { OUTPUT_DIR } from '@/lib/paths';

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: rawId } = await params;
  const run = await db.trainingRun.findUnique({ where: { id: parseInt(rawId) } });
  if (!run) return NextResponse.json({ error: 'not found' }, { status: 404 });
  return NextResponse.json(run);
}

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: rawId } = await params;
  const id = parseInt(rawId);
  const body = await req.json() as { label?: string; action?: 'pause' | 'resume' };

  if (body.action === 'pause' || body.action === 'resume') {
    const run = await db.trainingRun.findUnique({ where: { id } });
    if (!run) return NextResponse.json({ error: 'not found' }, { status: 404 });
    if (!run.pid) return NextResponse.json({ error: 'no pid' }, { status: 400 });

    if (body.action === 'pause') {
      try { process.kill(-run.pid, 'SIGSTOP'); } catch { /* process may have exited */ }
      await db.trainingRun.update({ where: { id }, data: { status: 'paused' } });
      return NextResponse.json({ ok: true, status: 'paused' });
    } else {
      try { process.kill(-run.pid, 'SIGCONT'); } catch { /* process may have exited */ }
      await db.trainingRun.update({ where: { id }, data: { status: 'running' } });
      return NextResponse.json({ ok: true, status: 'running' });
    }
  }

  const { label } = body;
  if (!label?.trim()) return NextResponse.json({ error: 'label required' }, { status: 400 });

  const run = await db.trainingRun.findUnique({ where: { id } });
  if (!run) return NextResponse.json({ error: 'not found' }, { status: 404 });

  const newSlug = label.trim().toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');
  const oldSlug = run.modelName;

  if (newSlug !== oldSlug) {
    for (const ext of ['.onnx', '.onnx.data']) {
      const oldPath = path.join(OUTPUT_DIR, `${oldSlug}${ext}`);
      const newPath = path.join(OUTPUT_DIR, `${newSlug}${ext}`);
      try {
        if (fs.existsSync(oldPath)) fs.renameSync(oldPath, newPath);
      } catch { /* ignore rename errors */ }
    }
    const oldDir = path.join(OUTPUT_DIR, oldSlug);
    const newDir = path.join(OUTPUT_DIR, newSlug);
    try {
      if (fs.existsSync(oldDir) && !fs.existsSync(newDir)) fs.renameSync(oldDir, newDir);
    } catch { /* ignore */ }
  }

  const updated = await db.trainingRun.update({
    where: { id },
    data: { label: label.trim(), modelName: newSlug },
  });

  return NextResponse.json(updated);
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: rawId } = await params;
  const id = parseInt(rawId);
  const run = await db.trainingRun.findUnique({ where: { id } });
  if (!run) return NextResponse.json({ error: 'not found' }, { status: 404 });
  if (run.status !== 'running' && run.status !== 'paused') {
    return NextResponse.json({ error: 'not active' }, { status: 400 });
  }

  if (run.pid) {
    try { process.kill(-run.pid, 'SIGTERM'); } catch { /* process may have already exited */ }
  }

  if (run.logFile) {
    try { fs.appendFileSync(run.logFile, '\n__FAILED__\n'); } catch { /* ignore */ }
  }

  await db.trainingRun.update({
    where: { id },
    data: { status: 'cancelled', finishedAt: new Date() },
  });

  return NextResponse.json({ ok: true });
}
