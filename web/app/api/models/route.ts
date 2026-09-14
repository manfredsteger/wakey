import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from '@/lib/paths';
import { fileMd5, getProvenance, type Provenance } from '@/lib/model-provenance';

export interface ModelFamily {
  wakeWord: string;
  esp32: { tflite: string; manifest: string | null; size: number; mtime: string; md5: string; provenance: Provenance } | null;
  wyoming: { onnx: string; data: string | null; size: number; mtime: string; md5: string; provenance: Provenance } | null;
}

export async function GET() {
  try {
    if (!fs.existsSync(OUTPUT_DIR)) return NextResponse.json([]);

    // Only scan top-level output/ — training intermediates live in subdirectories
    const entries = fs.readdirSync(OUTPUT_DIR, { withFileTypes: true })
      .filter(e => e.isFile());

    // Group by wake word stem (everything before the extension)
    const byWord: Record<string, string[]> = {};
    for (const e of entries) {
      if (!e.name.endsWith('.tflite') && !e.name.endsWith('.onnx') &&
          !e.name.endsWith('.onnx.data') && !e.name.endsWith('_manifest.json')) continue;
      // derive stem
      let stem = e.name
        .replace(/_manifest\.json$/, '')
        .replace(/\.onnx\.data$/, '')
        .replace(/\.onnx$/, '')
        .replace(/\.tflite$/, '');
      (byWord[stem] ??= []).push(e.name);
    }

    const families: ModelFamily[] = await Promise.all(Object.entries(byWord).map(async ([word, files]) => {
      const tflite = files.find(f => f.endsWith('.tflite')) ?? null;
      const manifest = files.find(f => f.endsWith('_manifest.json')) ?? null;
      const onnx = files.find(f => f.endsWith('.onnx')) ?? null;
      const data = files.find(f => f.endsWith('.onnx.data')) ?? null;

      const stat = (name: string | null) => {
        if (!name) return null;
        try { return fs.statSync(path.join(OUTPUT_DIR, name)); } catch { return null; }
      };

      const tfliteS = stat(tflite);
      const onnxS = stat(onnx);

      return {
        wakeWord: word,
        esp32: tflite && tfliteS ? {
          tflite, manifest,
          size: tfliteS.size + (stat(manifest)?.size ?? 0),
          mtime: tfliteS.mtime.toISOString(),
          md5: fileMd5(path.join(OUTPUT_DIR, tflite)),
          provenance: await getProvenance(word, 'esp32', tfliteS.mtime),
        } : null,
        wyoming: onnx && onnxS ? {
          onnx, data,
          size: onnxS.size + (stat(data)?.size ?? 0),
          mtime: onnxS.mtime.toISOString(),
          md5: fileMd5(path.join(OUTPUT_DIR, onnx)),
          provenance: await getProvenance(word, 'wyoming', onnxS.mtime),
        } : null,
      };
    }));

    families.sort((a, b) => {
      const ma = (a.esp32?.mtime ?? a.wyoming?.mtime) ?? '';
      const mb = (b.esp32?.mtime ?? b.wyoming?.mtime) ?? '';
      return mb.localeCompare(ma);
    });

    return NextResponse.json(families);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function DELETE(req: Request) {
  const { wakeWord, platform } = await req.json() as { wakeWord: string; platform: 'esp32' | 'wyoming' | 'all' };
  if (!wakeWord || wakeWord.includes('..')) return NextResponse.json({ error: 'Invalid' }, { status: 400 });

  const candidates: string[] = [];
  if (platform === 'esp32' || platform === 'all') {
    candidates.push(`${wakeWord}.tflite`, `${wakeWord}_manifest.json`);
  }
  if (platform === 'wyoming' || platform === 'all') {
    candidates.push(`${wakeWord}.onnx`, `${wakeWord}.onnx.data`);
  }

  for (const name of candidates) {
    const fp = path.join(OUTPUT_DIR, name);
    if (fs.existsSync(fp)) fs.unlinkSync(fp);
  }
  return NextResponse.json({ ok: true });
}
