'use client';

import { useEffect, useRef, useState } from 'react';
import { CheckCircle, XCircle, Pause, Play, StopCircle, Activity, Clock } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

type ProcessStatus = 'running' | 'paused' | 'done' | 'failed' | 'cancelled';

interface RunMeta {
  id: number;
  wakeWord: string;
  steps: number;
  platform: string;
  status: string;
}

interface Phase {
  index: number;
  total: number;
  label: string;
}

interface Metrics {
  step: number;
  accuracy: number;
  recall: number;
  loss: number;
}

type EvalState = 'training' | 'evaluating';

interface TrainingMonitorProps {
  runId: number;
  onDone?: () => void;
  onCancel?: () => void;
}

// Strip ANSI escape codes from log lines
function stripAnsi(s: string): string {
  return s.replace(/\x1B\[[0-9;]*[mGKHFJsu]/g, '').replace(/\[\d+m/g, '');
}

// Classify a log line for color coding
function lineClass(line: string): string {
  const t = line.trim();
  if (!t) return '';
  if (/^[╔╚║═]/.test(t)) return 'text-sky-400';
  if (t.startsWith('✓') || / ✓ /.test(t)) return 'text-emerald-400';
  if (/^\[(\d+)\/(\d+)\]/.test(t)) return 'text-white font-semibold';
  if (/INFO:absl:Step #\d+:/.test(t)) return 'text-amber-300';
  if (/^(Traceback|File "|raise |Error:|Exception:)/.test(t)) return 'text-red-400';
  if (/^WARNING/.test(t)) return 'text-yellow-500/60';
  if (/^INFO/.test(t)) return 'text-slate-500';
  if (t.startsWith('  ')) return 'text-slate-400';
  return 'text-slate-300';
}

// Should this line be hidden from the log?
function isNoisyLine(line: string): boolean {
  const t = line.trim();
  // Mini-Batch concatenated lines (huge, garbled)
  if (t.includes('Mini-Batch #') && t.includes('Validation Batch')) return true;
  // TF model summary table lines
  if (/^[┏┗┡┘└├│┃┩┼╇╈╉╊╋━─╔╚╠╦╩═╗╝╣┤]/.test(t)) return false; // keep box-drawing (banner)
  if (/^[┏┗┡┩├│┃].*\d+.*│/.test(t)) return true; // TF layer table rows
  if (/^│ \w/.test(t) && t.includes('│') && t.includes('(')) return true; // TF layer detail rows
  return false;
}

function formatEta(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

export function TrainingMonitor({ runId, onDone, onCancel }: TrainingMonitorProps) {
  const { t } = useI18n();
  const [lines, setLines] = useState<string[]>([]);
  const [processStatus, setProcessStatus] = useState<ProcessStatus>('running');
  const [runMeta, setRunMeta] = useState<RunMeta | null>(null);
  const [phase, setPhase] = useState<Phase | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [displayStep, setDisplayStep] = useState(0);
  const [evalState, setEvalState] = useState<EvalState>('training');
  const [eta, setEta] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const firstStepRef = useRef<{ step: number; time: number } | null>(null);
  const lastStepRef = useRef<{ step: number; time: number } | null>(null);
  // Stable refs for callbacks — prevents EventSource effect from re-running on every parent re-render
  const onDoneRef = useRef(onDone);
  const onCancelRef = useRef(onCancel);
  useEffect(() => { onDoneRef.current = onDone; });
  useEffect(() => { onCancelRef.current = onCancel; });
  // Refs for training-only rate (excludes eval overhead)
  const evalStateRef = useRef<EvalState>('training');
  const prevCheckpointTimeRef = useRef<number | null>(null); // when last training phase started
  const prevCheckpointStepRef = useRef(0);                  // step at last checkpoint
  const evalStartTimeRef = useRef<number | null>(null);     // when current eval started
  const trainingRateRef = useRef(0);                        // steps/ms (training only)

  const totalSteps = runMeta?.steps ?? 0;

  // Fetch run metadata
  useEffect(() => {
    fetch(`/api/train/${runId}`)
      .then(r => r.json())
      .then((run: RunMeta & { status: string }) => {
        setRunMeta(run);
        const s = run.status as ProcessStatus;
        if (s === 'paused' || s === 'cancelled' || s === 'done' || s === 'failed') {
          setProcessStatus(s);
        }
      })
      .catch(() => {});
  }, [runId]);

  // Parse a log line for phase / metrics / step info
  const parseLine = (line: string) => {
    // Phase header: [1/3] Description text
    const phaseM = line.match(/^\[(\d+)\/(\d+)\]\s+(.+)/);
    if (phaseM) {
      setPhase({ index: parseInt(phaseM[1]), total: parseInt(phaseM[2]), label: phaseM[3] });
      return;
    }

    // microWakeWord eval-interval step: INFO:absl:Step #N: rate R, accuracy A%, recall B%, ... cross entropy L
    // This line appears AFTER the eval pass — training for this interval is already done.
    const mwwM = line.match(/INFO:absl:Step #(\d+): rate [\d.]+, accuracy ([\d.]+)%, recall ([\d.]+)%, precision [\d.]+%, cross entropy ([\d.]+)/);
    if (mwwM) {
      const step = parseInt(mwwM[1]);
      const now = Date.now();
      // Calculate pure training rate: time from last INFO (= training start) to eval start
      if (prevCheckpointTimeRef.current !== null && evalStartTimeRef.current !== null) {
        const trainDuration = evalStartTimeRef.current - prevCheckpointTimeRef.current;
        const trainSteps = step - prevCheckpointStepRef.current;
        if (trainDuration > 0 && trainSteps > 0) {
          trainingRateRef.current = trainSteps / trainDuration;
        }
      }
      prevCheckpointStepRef.current = step;
      prevCheckpointTimeRef.current = now; // training starts now (after INFO + before next Mini-Batch)
      evalStartTimeRef.current = null;
      evalStateRef.current = 'training';

      const m: Metrics = { step, accuracy: parseFloat(mwwM[2]), recall: parseFloat(mwwM[3]), loss: parseFloat(mwwM[4]) };
      setMetrics(m);
      setCurrentStep(step);
      setDisplayStep(step); // snap to exact checkpoint value
      setEvalState('training');
      updateEta(step);
      return;
    }

    // Noisy Mini-Batch line: validation pass running between training steps.
    // Mini-Batch #N here = validation mini-batch, NOT a training step — don't update currentStep.
    if (line.includes('Mini-Batch #') && line.includes('Validation Batch')) {
      // First transition to evaluating: record when training ended
      if (evalStateRef.current !== 'evaluating') {
        evalStartTimeRef.current = Date.now();
        evalStateRef.current = 'evaluating';
        setEvalState('evaluating');
      }
      // Extract last validation metrics to keep the metrics panel live during eval
      const lastValM = line.match(/Accuracy = ([\d.]+); Recall = ([\d.]+); Precision = [\d.]+; Loss = ([\d.]+); Mini-Batch #\d+$/);
      if (lastValM) {
        setMetrics(prev => ({
          step: prev?.step ?? 0,
          accuracy: parseFloat(lastValM[1]) * 100,
          recall: parseFloat(lastValM[2]) * 100,
          loss: parseFloat(lastValM[3]),
        }));
      }
    }
  };

  const updateEta = (step: number) => {
    const now = Date.now();
    if (!firstStepRef.current) {
      firstStepRef.current = { step, time: now };
    }
    lastStepRef.current = { step, time: now };

    setRunMeta(meta => {
      if (!meta || !firstStepRef.current || !lastStepRef.current) return meta;
      const { step: s0, time: t0 } = firstStepRef.current;
      const { step: s1, time: t1 } = lastStepRef.current;
      if (s1 > s0 && t1 > t0) {
        const stepsPerMs = (s1 - s0) / (t1 - t0);
        const remSeconds = (meta.steps - s1) / stepsPerMs / 1000;
        if (remSeconds > 0) setEta(formatEta(remSeconds));
      }
      return meta;
    });
  };

  // EventSource streaming
  useEffect(() => {
    const es = new EventSource(`/api/train/${runId}/stream`);

    es.onmessage = (e) => {
      const data = e.data as string;
      if (data === '__DONE__') {
        setProcessStatus('done');
        es.close();
        onDoneRef.current?.();
        return;
      }
      if (data === '__FAILED__') {
        setProcessStatus(prev => prev === 'cancelled' ? 'cancelled' : 'failed');
        es.close();
        onDoneRef.current?.();
        return;
      }

      const clean = stripAnsi(data);
      parseLine(clean);

      // Only add non-noisy lines to the visible log
      if (!isNoisyLine(clean)) {
        setLines(prev => [...prev.slice(-600), clean]);
      }
    };

    es.onerror = () => { es.close(); };
    return () => es.close();
  }, [runId]); // onDone/onCancel intentionally via refs — inline props would cause infinite reconnect loop

  // Smooth step counter: advance displayStep between eval checkpoints using training-only rate
  useEffect(() => {
    if (processStatus !== 'running' || evalState !== 'training') return;
    const timer = setInterval(() => {
      if (trainingRateRef.current === 0 || prevCheckpointTimeRef.current === null) return;
      const elapsed = Date.now() - prevCheckpointTimeRef.current;
      const est = prevCheckpointStepRef.current + Math.floor(elapsed * trainingRateRef.current);
      setDisplayStep(prev => Math.min(Math.max(prev, est), totalSteps));
    }, 250);
    return () => clearInterval(timer);
  }, [processStatus, evalState, totalSteps]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines]);

  const handlePause = async () => {
    setProcessStatus('paused');
    await fetch(`/api/train/${runId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'pause' }),
    });
  };

  const handleResume = async () => {
    setProcessStatus('running');
    await fetch(`/api/train/${runId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'resume' }),
    });
  };

  const handleCancel = async () => {
    if (!window.confirm(t('training.cancelConfirm'))) return;
    setProcessStatus('cancelled');
    await fetch(`/api/train/${runId}`, { method: 'DELETE' });
    onCancelRef.current?.();
  };

  const isActive = processStatus === 'running' || processStatus === 'paused';
  const stepPct = totalSteps > 0 ? Math.min(100, (displayStep / totalSteps) * 100) : 0;
  const isTrainingPhase = !!(phase && phase.index === phase.total);
  // Show step progress when actively in training phase, or when a completed/failed run has step data
  const showStepProgress = isTrainingPhase || (!isActive && totalSteps > 0 && currentStep > 0);

  const barColor =
    processStatus === 'failed' || processStatus === 'cancelled' ? 'bg-red-500' :
    processStatus === 'paused' ? 'bg-amber-400' :
    'bg-emerald-500';

  return (
    <div className="space-y-4">
      {/* Phase stepper — always on top */}
      {phase && (
        <div className="flex items-center gap-1 overflow-x-auto pb-1">
          {Array.from({ length: phase.total }, (_, i) => {
            const idx = i + 1;
            const done = idx < phase.index;
            const active = idx === phase.index;
            return (
              <div key={idx} className="flex items-center gap-1 min-w-0">
                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium whitespace-nowrap ${
                  done ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400' :
                  active ? 'bg-emerald-500 text-white' :
                  'bg-slate-100 dark:bg-slate-800 text-slate-400'
                }`}>
                  {done ? <CheckCircle className="w-3 h-3 shrink-0" /> : (
                    <span className="w-3 h-3 flex items-center justify-center shrink-0 text-[10px] font-bold">{idx}</span>
                  )}
                  {active
                    ? phase.label.replace(/^Generating \d+ TTS samples?/, 'Generating TTS').replace(/\([\d,]+ steps\)/, '').trim()
                    : `Phase ${idx}`}
                </div>
                {idx < phase.total && (
                  <div className={`w-4 h-px shrink-0 ${done ? 'bg-emerald-400' : 'bg-slate-200 dark:bg-slate-700'}`} />
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Pre-training phases (1, 2, …): simple pulsing indicator */}
      {!isTrainingPhase && isActive && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            <Activity className="w-3 h-3 text-emerald-500 animate-pulse shrink-0" />
            <span className="truncate">
              {phase ? phase.label.replace(/\([\d,]+ steps\)/, '').trim() : t('training.running')}…
            </span>
          </div>
          <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-emerald-500 animate-pulse" style={{ width: '55%' }} />
          </div>
        </div>
      )}

      {/* Training phase: step-based progress */}
      {showStepProgress && (
        <div className="space-y-2">
          {/* Header row: icon + step count + Evaluating badge | ETA + % */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              {processStatus === 'running' && (
                <Activity className={`w-4 h-4 shrink-0 animate-pulse ${evalState === 'evaluating' ? 'text-amber-400' : 'text-emerald-500'}`} />
              )}
              {processStatus === 'paused' && <Activity className="w-4 h-4 shrink-0 text-amber-400" />}
              {processStatus === 'done' && <CheckCircle className="w-4 h-4 shrink-0 text-emerald-500" />}
              {(processStatus === 'failed' || processStatus === 'cancelled') && <XCircle className="w-4 h-4 shrink-0 text-red-500" />}
              <span className={`text-sm font-semibold tabular-nums ${
                processStatus === 'done' ? 'text-emerald-600 dark:text-emerald-400' :
                processStatus === 'failed' || processStatus === 'cancelled' ? 'text-red-500' :
                'text-slate-800 dark:text-slate-200'
              }`}>
                {processStatus === 'done'
                  ? `${totalSteps.toLocaleString()} / ${totalSteps.toLocaleString()} Steps`
                  : `Step ${displayStep.toLocaleString()} / ${totalSteps.toLocaleString()}`
                }
              </span>
              {evalState === 'evaluating' && processStatus === 'running' && (
                <span className="text-xs font-medium text-amber-500 dark:text-amber-400 shrink-0">· Evaluating…</span>
              )}
              {processStatus === 'paused' && (
                <span className="text-xs text-amber-500 dark:text-amber-400">{t('training.paused')}</span>
              )}
            </div>
            <div className="flex items-center gap-3 shrink-0 text-xs">
              {eta && processStatus === 'running' && (
                <span className="flex items-center gap-1 text-slate-500 dark:text-slate-400">
                  <Clock className="w-3 h-3" /> ~{eta}
                </span>
              )}
              <span className={`tabular-nums font-semibold ${
                processStatus === 'done' ? 'text-emerald-600 dark:text-emerald-400' :
                processStatus === 'failed' || processStatus === 'cancelled' ? 'text-red-500' :
                'text-slate-700 dark:text-slate-300'
              }`}>
                {Math.round(processStatus === 'done' ? 100 : stepPct)}%
              </span>
            </div>
          </div>
          {/* Step-based progress bar (0 → 100% of training steps) */}
          <div className="h-2.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${barColor} ${
                processStatus === 'running' && displayStep === 0 ? 'animate-pulse' : ''
              }`}
              style={{ width: `${processStatus === 'done' ? 100 : processStatus === 'running' && displayStep === 0 ? 3 : Math.max(stepPct, 0.3)}%` }}
            />
          </div>
        </div>
      )}

      {/* Failed / cancelled status line */}
      {!isActive && (processStatus === 'failed' || processStatus === 'cancelled') && (
        <div className="flex items-center gap-2 text-sm font-medium text-red-500">
          <XCircle className="w-4 h-4" />
          {processStatus === 'cancelled' ? t('training.cancelled') : t('training.failed')}
        </div>
      )}

      {/* Live metrics */}
      {metrics && (
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: 'Accuracy', value: `${metrics.accuracy.toFixed(1)}%`, color: 'text-emerald-500' },
            { label: 'Recall', value: `${metrics.recall.toFixed(1)}%`, color: 'text-sky-500' },
            { label: 'Loss', value: metrics.loss.toFixed(4), color: 'text-amber-500' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-slate-100 dark:bg-slate-800/60 rounded-lg px-3 py-2 text-center">
              <div className={`text-base font-bold tabular-nums ${color}`}>{value}</div>
              <div className="text-xs text-slate-400 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Log output */}
      <div ref={logRef} className="log-output">
        {lines.length === 0
          ? <span className="text-slate-600">{t('training.logEmpty')}</span>
          : lines.map((l, i) => (
            <div key={i} className={lineClass(l) || 'text-slate-300'}>
              {l || ' '}
            </div>
          ))
        }
      </div>

      {/* Controls */}
      {isActive && (
        <div className="flex items-center gap-2">
          <button
            onClick={processStatus === 'paused' ? handleResume : handlePause}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 hover:bg-amber-50 dark:hover:bg-amber-950/40 text-slate-600 dark:text-slate-300 hover:text-amber-600 dark:hover:text-amber-400 transition-colors"
          >
            {processStatus === 'paused' ? <><Play className="w-3.5 h-3.5" /> {t('training.resume')}</> : <><Pause className="w-3.5 h-3.5" /> {t('training.pause')}</>}
          </button>
          <button
            onClick={handleCancel}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 hover:bg-red-50 dark:hover:bg-red-950/40 text-slate-600 dark:text-slate-300 hover:text-red-600 dark:hover:text-red-400 transition-colors"
          >
            <StopCircle className="w-3.5 h-3.5" /> {t('training.cancel')}
          </button>
        </div>
      )}
    </div>
  );
}
