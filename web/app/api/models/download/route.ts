import fs from 'fs';
import path from 'path';
import JSZip from 'jszip';
import { OUTPUT_DIR } from '@/lib/paths';

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

export async function GET(req: Request) {
  const url = new URL(req.url);
  const wakeWord = url.searchParams.get('word');
  const platform = url.searchParams.get('platform') as 'esp32' | 'wyoming' | null;

  if (!wakeWord || !platform) return new Response('word and platform required', { status: 400 });
  if (wakeWord.includes('..')) return new Response('Invalid', { status: 400 });

  const zip = new JSZip();

  if (platform === 'esp32') {
    const tflite = path.join(OUTPUT_DIR, `${wakeWord}.tflite`);
    const manifest = path.join(OUTPUT_DIR, `${wakeWord}_manifest.json`);
    if (!fs.existsSync(tflite)) return new Response('Model not found', { status: 404 });
    zip.file(`${wakeWord}.tflite`, fs.readFileSync(tflite));
    if (fs.existsSync(manifest)) zip.file(`${wakeWord}_manifest.json`, fs.readFileSync(manifest));
    zip.file('README.md', readme(wakeWord, 'esp32'));
  } else {
    const onnx = path.join(OUTPUT_DIR, `${wakeWord}.onnx`);
    const data = path.join(OUTPUT_DIR, `${wakeWord}.onnx.data`);
    if (!fs.existsSync(onnx)) return new Response('Model not found', { status: 404 });
    zip.file(`${wakeWord}.onnx`, fs.readFileSync(onnx));
    if (fs.existsSync(data)) zip.file(`${wakeWord}.onnx.data`, fs.readFileSync(data));
    zip.file('README.md', readme(wakeWord, 'wyoming'));
  }

  const buf = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
  const arrayBuf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;

  return new Response(arrayBuf, {
    headers: {
      'Content-Type': 'application/zip',
      'Content-Disposition': `attachment; filename="${wakeWord}_${platform}.zip"`,
    },
  });
}
