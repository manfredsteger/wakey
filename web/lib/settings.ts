import fs from 'fs';
import path from 'path';
import { PROJECT_ROOT } from './paths';

const SETTINGS_FILE = path.join(PROJECT_ROOT, 'settings.json');

export interface Settings {
  haUrl: string;   // e.g. "http://homeassistant.local:8123"
  ttsLang: 'de' | 'en';  // TTS language for sample generation
}

const defaults: Settings = { haUrl: '', ttsLang: 'de' };

export function readSettings(): Settings {
  try {
    if (fs.existsSync(SETTINGS_FILE)) {
      return { ...defaults, ...JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf8')) };
    }
  } catch {}
  return { ...defaults };
}

export function writeSettings(patch: Partial<Settings>): Settings {
  const updated = { ...readSettings(), ...patch };
  fs.writeFileSync(SETTINGS_FILE, JSON.stringify(updated, null, 2));
  return updated;
}
