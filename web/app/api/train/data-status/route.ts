import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { DATA_DIR } from '@/lib/paths';

export async function GET() {
  const acav = fs.existsSync(
    path.join(DATA_DIR, 'openwakeword_features_ACAV100M_2000_hrs_16bit.npy')
  );
  return NextResponse.json({ acav });
}
