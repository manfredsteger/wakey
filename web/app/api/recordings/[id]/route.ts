import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from '@/lib/paths';

function trainDir(modelName: string) {
  return path.join(OUTPUT_DIR, modelName.replace(/\s+/g, '_'), 'positive_train');
}

function speakerSlug(speaker: string) {
  return speaker.toLowerCase().replace(/\s+/g, '_');
}

function parseId(raw: string): { modelName: string; speaker: string } | null {
  const [modelName, speaker] = decodeURIComponent(raw).split('::');
  if (!modelName || !speaker) return null;
  return { modelName, speaker };
}

// GET: return individual file list for a speaker
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const parsed = parseId(id);
  if (!parsed) return NextResponse.json({ error: 'Invalid id' }, { status: 400 });

  const { modelName, speaker } = parsed;
  const slug = speakerSlug(speaker);
  const dir = trainDir(modelName);

  if (!fs.existsSync(dir)) return NextResponse.json({ files: [] });

  const files = fs.readdirSync(dir)
    .filter(f => f.startsWith(`real_${slug}_`) && f.endsWith('.wav'))
    .sort();

  return NextResponse.json({ id, speaker, modelName, files });
}

// PATCH: rename speaker (renames all files)
export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const parsed = parseId(id);
  if (!parsed) return NextResponse.json({ error: 'Invalid id' }, { status: 400 });

  const { newSpeaker } = await req.json() as { newSpeaker: string };
  if (!newSpeaker?.trim()) return NextResponse.json({ error: 'newSpeaker required' }, { status: 400 });

  const { modelName, speaker } = parsed;
  const oldSlug = speakerSlug(speaker);
  const newSlug = speakerSlug(newSpeaker.trim());

  if (oldSlug === newSlug) return NextResponse.json({ id, speaker: newSpeaker.trim() });

  const dir = trainDir(modelName);
  if (!fs.existsSync(dir)) return NextResponse.json({ error: 'not found' }, { status: 404 });

  const toRename = fs.readdirSync(dir)
    .filter(f => f.startsWith(`real_${oldSlug}_`) && f.endsWith('.wav'));

  for (const file of toRename) {
    const newName = file.replace(`real_${oldSlug}_`, `real_${newSlug}_`);
    fs.renameSync(path.join(dir, file), path.join(dir, newName));
  }

  const newId = `${modelName}::${newSpeaker.trim().toLowerCase().replace(/\s+/g, '_').replace(/_/g, ' ')}`;
  return NextResponse.json({ id: newId, speaker: newSpeaker.trim(), renamed: toRename.length });
}

// DELETE: all recordings for speaker, or specific files via body { files: string[] }
export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const parsed = parseId(id);
  if (!parsed) return NextResponse.json({ error: 'Invalid id' }, { status: 400 });

  const { modelName, speaker } = parsed;
  const slug = speakerSlug(speaker);
  const dir = trainDir(modelName);

  if (!fs.existsSync(dir)) return NextResponse.json({ ok: true, deleted: 0 });

  // Check if specific files requested
  let specificFiles: string[] | null = null;
  try {
    const body = await req.json() as { files?: string[] };
    if (Array.isArray(body.files) && body.files.length > 0) specificFiles = body.files;
  } catch { /* no body = delete all */ }

  const candidates = specificFiles
    ? specificFiles
    : fs.readdirSync(dir).filter(f => f.startsWith(`real_${slug}_`) && f.endsWith('.wav'));

  let deleted = 0;
  for (const file of candidates) {
    const filePath = path.join(dir, path.basename(file));
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      deleted++;
    }
  }

  return NextResponse.json({ ok: true, deleted });
}
