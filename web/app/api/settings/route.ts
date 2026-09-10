import { NextResponse } from 'next/server';
import { readSettings, writeSettings } from '@/lib/settings';

export async function GET() {
  return NextResponse.json(readSettings());
}

export async function PUT(req: Request) {
  const body = await req.json() as { haUrl?: string; ttsLang?: 'de' | 'en' };
  return NextResponse.json(writeSettings(body));
}
