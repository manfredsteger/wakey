import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { OUTPUT_DIR } from '@/lib/paths';

interface Group {
  id: string;
  name: string;
  models: string[];   // model file names, e.g. ["hey_dobbi.tflite"]
  notes: string;
}

const GROUPS_FILE = path.join(OUTPUT_DIR, '_groups.json');

function load(): Group[] {
  try {
    return (JSON.parse(fs.readFileSync(GROUPS_FILE, 'utf8')) as { groups: Group[] }).groups ?? [];
  } catch {
    return [];
  }
}

function save(groups: Group[]) {
  fs.mkdirSync(path.dirname(GROUPS_FILE), { recursive: true });
  fs.writeFileSync(GROUPS_FILE, JSON.stringify({ groups }, null, 2));
}

export async function GET() {
  return NextResponse.json(load());
}

export async function POST(req: Request) {
  const { name, notes } = await req.json() as { name: string; notes?: string };
  if (!name?.trim()) return NextResponse.json({ error: 'name required' }, { status: 400 });
  const groups = load();
  const group: Group = { id: crypto.randomUUID(), name: name.trim(), models: [], notes: notes?.trim() ?? '' };
  groups.push(group);
  save(groups);
  return NextResponse.json(group);
}

export async function PATCH(req: Request) {
  const { id, name, notes, models } = await req.json() as { id: string; name?: string; notes?: string; models?: string[] };
  if (!id) return NextResponse.json({ error: 'id required' }, { status: 400 });
  const groups = load().map(g =>
    g.id === id
      ? {
          ...g,
          ...(name !== undefined ? { name: name.trim() } : {}),
          ...(notes !== undefined ? { notes: notes.trim() } : {}),
          ...(models !== undefined ? { models } : {}),
        }
      : g
  );
  save(groups);
  return NextResponse.json({ ok: true });
}

export async function DELETE(req: Request) {
  const { id } = await req.json() as { id: string };
  if (!id) return NextResponse.json({ error: 'id required' }, { status: 400 });
  save(load().filter(g => g.id !== id));
  return NextResponse.json({ ok: true });
}
