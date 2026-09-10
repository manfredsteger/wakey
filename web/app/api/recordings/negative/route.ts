import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import os from 'os';
import { OUTPUT_DIR } from '@/lib/paths';

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const wakeWord = searchParams.get('wakeWord') ?? '';
  const modelName = wakeWord.toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');
  const negDir = path.join(OUTPUT_DIR, modelName, 'negative_train');

  try {
    if (!fs.existsSync(negDir)) return NextResponse.json([]);
    const groups: Record<string, number> = {};
    for (const f of fs.readdirSync(negDir)) {
      const m = f.match(/^real_neg_(.+?)_\d+\.wav$/);
      if (!m) continue;
      const phrase = m[1].replace(/_/g, ' ');
      groups[phrase] = (groups[phrase] ?? 0) + 1;
    }
    return NextResponse.json(Object.entries(groups).map(([phrase, count]) => ({ phrase, count })));
  } catch {
    return NextResponse.json([]);
  }
}

export async function POST(req: Request) {
  const form = await req.formData();
  const audio = form.get('audio') as File | null;
  // Recorder appends wakeWord={phrase} first, then extraFields appends wakeWord={actualModel}.
  // getAll returns both — we want the last one (the actual model name from extraFields).
  const wakeWordValues = form.getAll('wakeWord') as string[];
  const wakeWord = (wakeWordValues.at(-1) ?? wakeWordValues[0]) as string;
  const phrase = form.get('phrase') as string;

  if (!audio || !wakeWord || !phrase)
    return NextResponse.json({ error: 'Missing fields' }, { status: 400 });

  const modelName = wakeWord.toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');
  const phraseSlug = phrase.toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');
  const negDir = path.join(OUTPUT_DIR, modelName, 'negative_train');
  fs.mkdirSync(negDir, { recursive: true });

  const existing = fs.readdirSync(negDir).filter(f => f.startsWith(`real_neg_${phraseSlug}_`));
  const idx = existing.length + 1;
  const outPath = path.join(negDir, `real_neg_${phraseSlug}_${String(idx).padStart(3, '0')}.wav`);

  const tmpIn = path.join(os.tmpdir(), `neg_in_${Date.now()}.wav`);
  fs.writeFileSync(tmpIn, Buffer.from(await audio.arrayBuffer()));
  try {
    execFileSync('ffmpeg', [
      '-y', '-i', tmpIn,
      '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', outPath,
    ], { stdio: 'ignore' });
    fs.unlinkSync(tmpIn);
  } catch {
    fs.renameSync(tmpIn, outPath);
  }

  return NextResponse.json({ ok: true, file: path.basename(outPath) });
}

export async function DELETE(req: Request) {
  const { wakeWord, phrase } = await req.json() as { wakeWord: string; phrase: string };
  const modelName = wakeWord.toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');
  const phraseSlug = phrase.toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');
  const negDir = path.join(OUTPUT_DIR, modelName, 'negative_train');
  try {
    for (const f of fs.readdirSync(negDir)) {
      if (f.startsWith(`real_neg_${phraseSlug}_`)) fs.unlinkSync(path.join(negDir, f));
    }
  } catch { /* ignore */ }
  return NextResponse.json({ ok: true });
}
