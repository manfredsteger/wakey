import fs from 'fs';
import path from 'path';
import JSZip from 'jszip';
import { OUTPUT_DIR } from '@/lib/paths';
import { fileMd5, getProvenance, type Provenance } from '@/lib/model-provenance';

function readme(wakeWord: string, platform: 'esp32' | 'wyoming'): string {
  const display = wakeWord.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  if (platform === 'esp32') {
    return `# ${display} — ESP32 / ESPHome (microWakeWord)

## Enthaltene Dateien
- \`${wakeWord}.tflite\` — Quantisiertes TFLite-Modell für ESP32
- \`${wakeWord}_manifest.json\` — ESPHome-Manifest

## Installation

1. Beide Dateien in das HA-Media-Verzeichnis kopieren:
   \`\`\`
   scp ${wakeWord}.tflite ${wakeWord}_manifest.json root@homeassistant.local:/config/www/
   \`\`\`

2. ESPHome-Konfiguration:
   \`\`\`yaml
   micro_wake_word:
     models:
       - model: http://homeassistant.local:8123/local/${wakeWord}.tflite
   \`\`\`

## Hinweise
- Kompatibel mit ESPHome >= 2024.2
- Benötigt ESP32-S3 oder ESP32 mit >= 520 KB SRAM
- Erkennt: **${display}**
`;
  }
  return `# ${display} — Wyoming / openWakeWord

## Enthaltene Dateien
- \`${wakeWord}.onnx\` — ONNX-Modell
- \`${wakeWord}.onnx.data\` — Externe Gewichte (muss zusammen mit .onnx liegen)

## Installation

1. **Beide Dateien** gemeinsam kopieren:
   \`\`\`bash
   scp ${wakeWord}.onnx ${wakeWord}.onnx.data root@homeassistant.local:/share/openwakeword/
   \`\`\`

2. Wyoming-openWakeWord-Dienst neu starten
   (HA → Einstellungen → Add-ons → openWakeWord → Neustart)

3. Pipeline konfigurieren:
   Einstellungen → Sprachassistent → Pipeline → Wake Word: **${display}**

## Hinweise
- Kompatibel mit openWakeWord >= 0.5
- .onnx und .onnx.data müssen immer im gleichen Verzeichnis liegen
- Erkennt: **${display}**
`;
}

function provenanceReadme(prov: Provenance): string {
  const run = prov.producedBy ? `Run #${prov.producedBy.id}, fertig ${prov.producedBy.finishedAt}` : 'unbekannter Lauf (kein passender Eintrag in der Trainings-DB)';
  return `
## Herkunft
- Erzeugt von: **${run}**
- Aktualität: ${prov.stale ? `**VERALTET** (${prov.staleReason})` : 'neuestes trainiertes Modell'}
- Details und MD5-Summen: siehe \`BUILD_INFO.txt\`
`;
}

function buildInfo(wakeWord: string, platform: 'esp32' | 'wyoming', files: { name: string; path: string }[], prov: Provenance): string {
  const lines = [
    `Wake word:   ${wakeWord}`,
    `Platform:    ${platform}`,
    `Exported:    ${new Date().toISOString()}`,
    `Produced by: ${prov.producedBy ? `Run #${prov.producedBy.id} (finished ${prov.producedBy.finishedAt})` : 'unknown run (no matching training run in DB)'}`,
    `Latest run:  ${prov.latestRun ? `Run #${prov.latestRun.id} · ${prov.latestRun.status} · started ${prov.latestRun.createdAt}` : 'none'}`,
    `Stale:       ${prov.stale ? `YES (${prov.staleReason})` : 'no — this is the newest trained model'}`,
    '',
    'Files:',
  ];
  for (const f of files) {
    const st = fs.statSync(f.path);
    lines.push(`  ${f.name}  ${st.size} bytes  mtime ${st.mtime.toISOString()}  md5 ${fileMd5(f.path)}`);
  }
  return lines.join('\n') + '\n';
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const wakeWord = url.searchParams.get('word');
  const platform = url.searchParams.get('platform') as 'esp32' | 'wyoming' | null;
  const force = url.searchParams.get('force') === '1';

  if (!wakeWord || !platform) return new Response('word and platform required', { status: 400 });
  if (wakeWord.includes('..')) return new Response('Invalid', { status: 400 });

  const zip = new JSZip();
  const primary = path.join(OUTPUT_DIR, platform === 'esp32' ? `${wakeWord}.tflite` : `${wakeWord}.onnx`);
  if (!fs.existsSync(primary)) return new Response('Model not found', { status: 404 });

  // Never hand out a file that cannot be the newest training result without saying so.
  const prov = await getProvenance(wakeWord, platform, fs.statSync(primary).mtime);
  if (prov.stale && !force) {
    return Response.json({
      error: 'stale_model',
      reason: prov.staleReason,
      message: prov.staleReason === 'training_running'
        ? 'Ein Training für dieses Modell läuft gerade – die Datei wird gleich überschrieben.'
        : prov.staleReason === 'newer_run_no_file'
          ? 'Ein neuerer Trainingslauf ist fertig, aber die Datei auf der Platte ist älter. Bitte Lauf/Log prüfen.'
          : 'Der letzte Trainingslauf ist fehlgeschlagen; diese Datei stammt aus einem älteren Lauf.',
      provenance: prov,
    }, { status: 409 });
  }
  const bundled: { name: string; path: string }[] = [];

  if (platform === 'esp32') {
    const tflite = path.join(OUTPUT_DIR, `${wakeWord}.tflite`);
    const manifest = path.join(OUTPUT_DIR, `${wakeWord}_manifest.json`);
    zip.file(`${wakeWord}.tflite`, fs.readFileSync(tflite));
    bundled.push({ name: `${wakeWord}.tflite`, path: tflite });
    if (fs.existsSync(manifest)) { zip.file(`${wakeWord}_manifest.json`, fs.readFileSync(manifest)); bundled.push({ name: `${wakeWord}_manifest.json`, path: manifest }); }
    zip.file('README.md', readme(wakeWord, 'esp32') + provenanceReadme(prov));
  } else {
    const onnx = path.join(OUTPUT_DIR, `${wakeWord}.onnx`);
    const data = path.join(OUTPUT_DIR, `${wakeWord}.onnx.data`);
    zip.file(`${wakeWord}.onnx`, fs.readFileSync(onnx));
    bundled.push({ name: `${wakeWord}.onnx`, path: onnx });
    if (fs.existsSync(data)) { zip.file(`${wakeWord}.onnx.data`, fs.readFileSync(data)); bundled.push({ name: `${wakeWord}.onnx.data`, path: data }); }
    zip.file('README.md', readme(wakeWord, 'wyoming') + provenanceReadme(prov));
  }
  zip.file('BUILD_INFO.txt', buildInfo(wakeWord, platform, bundled, prov));

  const buf = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
  const arrayBuf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;

  return new Response(arrayBuf, {
    headers: {
      'Content-Type': 'application/zip',
      'Content-Disposition': `attachment; filename="${wakeWord}_${platform}${prov.producedBy ? `_run${prov.producedBy.id}` : ''}.zip"`,
      'X-Model-Run': prov.producedBy ? String(prov.producedBy.id) : 'unknown',
      'X-Model-MD5': fileMd5(primary),
      'X-Model-Stale': prov.stale ? '1' : '0',
      'Cache-Control': 'no-store',
    },
  });
}
