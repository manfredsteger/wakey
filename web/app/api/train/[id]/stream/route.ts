import fs from 'fs';
import { db } from '@/lib/db';

function isPidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id: rawId } = await params;
  const run = await db.trainingRun.findUnique({ where: { id: parseInt(rawId) } });
  if (!run?.logFile) return new Response('Not found', { status: 404 });

  const encoder = new TextEncoder();
  let timerId: ReturnType<typeof setTimeout> | null = null;

  const stream = new ReadableStream({
    async start(controller) {
      let offset = 0;
      let done = false;

      const send = (line: string) => {
        try { controller.enqueue(encoder.encode(`data: ${line}\n\n`)); } catch { /* client gone */ }
      };

      const close = () => {
        if (timerId) { clearTimeout(timerId); timerId = null; }
        try { controller.close(); } catch { /* already closed */ }
      };

      // Cancel polling when the client disconnects
      req.signal.addEventListener('abort', close, { once: true });

      const tick = async () => {
        if (done || req.signal.aborted) { close(); return; }
        try {
          const stat = fs.statSync(run.logFile!);
          if (stat.size > offset) {
            const buf = Buffer.alloc(stat.size - offset);
            const fd = fs.openSync(run.logFile!, 'r');
            fs.readSync(fd, buf, 0, buf.length, offset);
            fs.closeSync(fd);
            offset = stat.size;

            for (const line of buf.toString('utf8').split('\n')) {
              const trimmed = line.trim();
              if (!trimmed) continue;
              if (trimmed === '__DONE__') {
                send('__DONE__');
                done = true;
                // Ensure DB is up to date (covers hot-reload close-handler loss)
                try {
                  await db.trainingRun.update({
                    where: { id: run.id },
                    data: { status: 'done', finishedAt: new Date() },
                  });
                } catch { /* ignore */ }
                break;
              }
              if (trimmed === '__FAILED__') {
                send('__FAILED__');
                done = true;
                try {
                  await db.trainingRun.update({
                    where: { id: run.id },
                    data: { status: 'failed', finishedAt: new Date() },
                  });
                } catch { /* ignore */ }
                break;
              }
              send(trimmed);
            }
          }

          if (!done) {
            const fresh = await db.trainingRun.findUnique({ where: { id: run.id } });
            if (fresh && fresh.status !== 'running' && fresh.status !== 'paused') {
              done = true;
            } else if (fresh && fresh.pid && !isPidAlive(fresh.pid)) {
              // Process is dead but DB still shows running — close handler was lost (dev restart).
              // Determine outcome from log content written by train.py itself.
              let logContent = '';
              try { logContent = fs.readFileSync(run.logFile!, 'utf8'); } catch { /* ignore */ }
              const finalStatus = logContent.includes('__DONE__') ? 'done' : 'failed';
              send(finalStatus === 'done' ? '__DONE__' : '__FAILED__');
              done = true;
              try {
                await db.trainingRun.update({
                  where: { id: run.id },
                  data: { status: finalStatus, finishedAt: new Date() },
                });
              } catch { /* ignore */ }
            }
          }
        } catch { /* stat/read error — process may have ended */ }

        if (done || req.signal.aborted) { close(); return; }
        timerId = setTimeout(tick, 500);
      };

      timerId = setTimeout(tick, 200);
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}
