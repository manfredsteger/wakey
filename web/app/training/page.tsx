'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Zap, Mic, Info, Users, Server, Cpu, Layers, CheckCircle2, AlertCircle, Volume2, ShieldAlert } from 'lucide-react';
import { Header } from '@/components/header';
import { TrainingMonitor } from '@/components/training-monitor';
import { StatusBadge } from '@/components/status-badge';
import { useI18n } from '@/lib/i18n';
import { formatDate } from '@/lib/utils';
import { cn } from '@/lib/utils';

type Preset = 'low' | 'mid' | 'max';
const PRESETS: Record<Preset, { samples: number; steps: number; full: boolean }> = {
  low: { samples: 200, steps: 2000, full: false },
  mid: { samples: 1000, steps: 10000, full: false },
  max: { samples: 2000, steps: 25000, full: true },
};
const PRESET_STORAGE_KEY = 'training_preset';

interface Run {
  id: number;
  wakeWord: string;
  label: string | null;
  samples: number;
  steps: number;
  fullMode: boolean;
  hasRealVoice: boolean;
  status: string;
  platform: string;
  createdAt: string;
}

interface Speaker {
  id: string;
  speaker: string;
  wakeWord: string;
  count: number;
}

function toSlug(s: string) {
  return s.toLowerCase().replace(/[\s,!.]+/g, '_').replace(/_+/g, '_');
}

export default function TrainingPage() {
  const { t } = useI18n();
  const [wakeWord, setWakeWord] = useState('Hey Dobbi');
  const [samples, setSamples] = useState(2000);
  const [steps, setSteps] = useState(25000);
  const [full, setFull] = useState(true);
  const [preset, setPreset] = useState<Preset>('max');
  const [acavReady, setAcavReady] = useState<boolean | null>(null);
  const [platform, setPlatform] = useState<'openWakeWord' | 'microWakeWord' | 'both'>('microWakeWord');
  const [activeRun, setActiveRun] = useState<number | null>(null);
  const [recentRuns, setRecentRuns] = useState<Run[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [matchingSpeakers, setMatchingSpeakers] = useState<Speaker[]>([]);
  const [bgCount, setBgCount] = useState(0);
  const [negCount, setNegCount] = useState(0);

  const loadMatchingSpeakers = useCallback(async (word: string) => {
    const res = await fetch('/api/recordings');
    if (!res.ok) return;
    const all: Speaker[] = await res.json();
    setMatchingSpeakers(all.filter(s => toSlug(s.wakeWord) === toSlug(word)));
  }, []);

  const loadNegCount = useCallback(async (word: string) => {
    const res = await fetch(`/api/recordings/negative?wakeWord=${encodeURIComponent(word)}`);
    if (!res.ok) return;
    const phrases: { phrase: string; count: number }[] = await res.json();
    setNegCount(phrases.reduce((s, p) => s + p.count, 0));
  }, []);

  useEffect(() => {
    loadMatchingSpeakers(wakeWord);
    loadNegCount(wakeWord);
  }, [wakeWord, loadMatchingSpeakers, loadNegCount]);

  useEffect(() => {
    fetch('/api/recordings/background')
      .then(r => r.json())
      .then(({ count }: { count: number }) => setBgCount(count))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(PRESET_STORAGE_KEY) as Preset | null;
    const p: Preset = (saved && saved in PRESETS) ? saved : 'max';
    setPreset(p);
    setSamples(PRESETS[p].samples);
    setSteps(PRESETS[p].steps);
    setFull(PRESETS[p].full);
  }, []);

  const applyPreset = (p: Preset) => {
    setPreset(p);
    setSamples(PRESETS[p].samples);
    setSteps(PRESETS[p].steps);
    setFull(PRESETS[p].full);
    localStorage.setItem(PRESET_STORAGE_KEY, p);
  };

  useEffect(() => {
    fetch('/api/train/data-status').then(r => r.json()).then(({ acav }: { acav: boolean }) => {
      setAcavReady(acav);
    }).catch(() => setAcavReady(false));
  }, []);

  const loadHistory = async () => {
    const res = await fetch('/api/train');
    if (res.ok) {
      const runs: Run[] = await res.json();
      setRecentRuns(runs);
      // Auto-restore monitor if a run is still active or paused
      if (activeRun === null) {
        const active = runs.find(r => r.status === 'running' || r.status === 'paused');
        if (active) setActiveRun(active.id);
      }
    }
  };

  // Load history on mount and auto-restore any running training
  useEffect(() => { loadHistory(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const startTraining = async () => {
    if (!wakeWord.trim() || isStarting) return;
    setIsStarting(true);
    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wakeWord: wakeWord.trim(), samples, steps, full, platform }),
      });
      const data = await res.json();
      if (res.status === 409) {
        // Another training is already running — jump to its monitor
        setActiveRun(data.id);
        loadHistory();
        return;
      }
      if (data.id) {
        setActiveRun(data.id);
        loadHistory();
      }
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Header />
      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Zap className="w-6 h-6 text-emerald-500" />
            {t('training.title')}
          </h1>
        </div>

        <div className={activeRun ? 'space-y-4' : 'grid grid-cols-1 lg:grid-cols-2 gap-6'}>
          {/* Config form — hidden when monitor is active */}
          <div className={`card p-6 space-y-5 ${activeRun ? 'hidden' : ''}`}>
            {/* Platform selector */}
            <div>
              <label className="label">{t('training.platform')}</label>
              <div className="grid grid-cols-3 gap-2 mt-1">
                {([
                  { value: 'microWakeWord', icon: Cpu, label: t('training.platformMWW'), hint: t('training.platformMWWHint') },
                  { value: 'openWakeWord', icon: Server, label: t('training.platformOWW'), hint: t('training.platformOWWHint') },
                  { value: 'both', icon: Layers, label: t('training.platformBoth'), hint: t('training.platformBothHint') },
                ] as const).map(({ value, icon: Icon, label, hint }) => (
                  <button
                    key={value}
                    type="button"
                    disabled={!!activeRun}
                    onClick={() => setPlatform(value)}
                    className={`flex flex-col items-start gap-1 p-3 rounded-xl border-2 text-left transition-all ${
                      platform === value
                        ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/40'
                        : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                    } ${!!activeRun ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                  >
                    <div className="flex items-center gap-2">
                      <Icon className={`w-4 h-4 ${platform === value ? 'text-emerald-500' : 'text-slate-400'}`} />
                      <span className={`text-sm font-semibold ${platform === value ? 'text-emerald-700 dark:text-emerald-400' : 'text-slate-700 dark:text-slate-300'}`}>{label}</span>
                    </div>
                    <span className="text-xs text-slate-500 dark:text-slate-400 leading-tight">{hint}</span>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="label">{t('training.wakeWord')}</label>
              <input
                className="input"
                value={wakeWord}
                onChange={e => setWakeWord(e.target.value)}
                placeholder={t('training.wakeWordPlaceholder')}
                disabled={!!activeRun}
              />
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">{t('training.wakeWordHint')}</p>

              {/* Training data badges */}
              {(matchingSpeakers.length > 0 || bgCount > 0 || negCount > 0) && (
                <div className="mt-2 space-y-1.5">
                  {matchingSpeakers.length > 0 && (
                    <div className="p-2.5 rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-100 dark:border-blue-900 flex items-start gap-2">
                      <Users className="w-3.5 h-3.5 text-blue-500 shrink-0 mt-0.5" />
                      <div className="text-xs text-blue-700 dark:text-blue-300">
                        <span className="font-medium">{t('training.realVoicesIncluded')}: </span>
                        {matchingSpeakers.map((s, i) => (
                          <span key={s.id}>
                            {i > 0 && ', '}
                            <span className="font-semibold">{s.speaker}</span>
                            <span className="opacity-70"> ({s.count}×)</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {bgCount > 0 && (
                    <div className="p-2.5 rounded-lg bg-orange-50 dark:bg-orange-950/40 border border-orange-100 dark:border-orange-900 flex items-center gap-2">
                      <Volume2 className="w-3.5 h-3.5 text-orange-500 shrink-0" />
                      <p className="text-xs text-orange-700 dark:text-orange-300">
                        <span className="font-medium">{t('training.bgSoundsIncluded')}: </span>
                        <span className="font-semibold">{bgCount}</span>
                        <span className="opacity-70"> Clips</span>
                      </p>
                    </div>
                  )}
                  {negCount > 0 && (
                    <div className="p-2.5 rounded-lg bg-red-50 dark:bg-red-950/40 border border-red-100 dark:border-red-900 flex items-center gap-2">
                      <ShieldAlert className="w-3.5 h-3.5 text-red-500 shrink-0" />
                      <p className="text-xs text-red-700 dark:text-red-300">
                        <span className="font-medium">{t('training.negativesIncluded')}: </span>
                        <span className="font-semibold">{negCount}</span>
                        <span className="opacity-70"> Aufnahmen</span>
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Quality presets */}
            <div>
              <label className="label">{t('training.preset')}</label>
              <div className="grid grid-cols-3 gap-2 mt-1">
                {([
                  { value: 'low' as Preset, label: t('training.presetLow'), hint: t('training.presetLowHint') },
                  { value: 'mid' as Preset, label: t('training.presetMid'), hint: t('training.presetMidHint') },
                  { value: 'max' as Preset, label: t('training.presetMax'), hint: t('training.presetMaxHint') },
                ]).map(({ value, label, hint }) => (
                  <button
                    key={value}
                    type="button"
                    disabled={!!activeRun}
                    onClick={() => applyPreset(value)}
                    className={cn(
                      'flex flex-col items-start gap-0.5 px-3 py-2.5 rounded-xl border-2 text-left transition-all',
                      preset === value
                        ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/40'
                        : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600',
                      !!activeRun ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                    )}
                  >
                    <span className={cn('text-sm font-semibold', preset === value ? 'text-emerald-700 dark:text-emerald-400' : 'text-slate-700 dark:text-slate-300')}>{label}</span>
                    <span className="text-xs text-slate-400 leading-tight">{hint}</span>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="label">{t('training.samples')}: <span className="text-emerald-600 dark:text-emerald-400 font-bold">{samples}</span></label>
              <input
                type="range" min={30} max={2000} step={50}
                value={samples}
                onChange={e => setSamples(Number(e.target.value))}
                disabled={!!activeRun}
                className="w-full accent-emerald-500"
              />
              <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                <span>30 (Test)</span><span>2000 (Produktion)</span>
              </div>
            </div>

            <div>
              <label className="label">{t('training.steps')}: <span className="text-emerald-600 dark:text-emerald-400 font-bold">{steps.toLocaleString()}</span></label>
              <input
                type="range" min={50} max={25000} step={500}
                value={steps}
                onChange={e => setSteps(Number(e.target.value))}
                disabled={!!activeRun}
                className="w-full accent-emerald-500"
              />
              <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                <span>50 (Test)</span><span>25000 (Max)</span>
              </div>
            </div>

            <div>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox" checked={full} onChange={e => setFull(e.target.checked)}
                  disabled={!!activeRun}
                  className="accent-emerald-500"
                />
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{t('training.fullMode')}</span>
              </label>
              {acavReady !== null && (
                <div className={`mt-1.5 flex items-center gap-1.5 text-xs ml-6 ${acavReady ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`}>
                  {acavReady
                    ? <><CheckCircle2 className="w-3.5 h-3.5 shrink-0" />{t('training.fullModeReady')}</>
                    : <><AlertCircle className="w-3.5 h-3.5 shrink-0" />{t('training.fullModeNotReady')}</>
                  }
                </div>
              )}
            </div>

            {!activeRun ? (
              <button onClick={startTraining} disabled={isStarting || !wakeWord.trim()} className="btn-primary w-full">
                {isStarting ? t('common.loading') : t('training.start')}
              </button>
            ) : (
              <button onClick={() => { setActiveRun(null); loadHistory(); }} className="btn-secondary w-full">
                ← {t('training.newTraining')}
              </button>
            )}

            {/* Tip */}
            <div className="flex items-start gap-2 p-3 bg-emerald-50 dark:bg-emerald-950/40 rounded-lg border border-emerald-100 dark:border-emerald-900">
              <Info className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
              <Link href="/recordings" className="text-xs text-emerald-700 dark:text-emerald-300 hover:underline">
                {t('training.realVoiceHint')}
              </Link>
            </div>
          </div>

          {/* Monitor — placeholder when no run, full card when active */}
          {activeRun ? (
            <div className="card p-6 space-y-2">
              {/* Header row with wake word + back button */}
              <div className="flex items-start justify-between gap-4 pb-2 border-b border-slate-100 dark:border-slate-800">
                <div className="min-w-0">
                  <h2 className="font-semibold text-slate-900 dark:text-white truncate">
                    &ldquo;{recentRuns.find(r => r.id === activeRun)?.label ?? recentRuns.find(r => r.id === activeRun)?.wakeWord ?? '…'}&rdquo;
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {recentRuns.find(r => r.id === activeRun)?.platform === 'microWakeWord' ? 'ESP32 · microWakeWord' : recentRuns.find(r => r.id === activeRun)?.platform === 'both' ? 'ESP32 + Wyoming · Beide' : 'Wyoming · openWakeWord'}
                    {' · '}
                    {(recentRuns.find(r => r.id === activeRun)?.steps ?? 0).toLocaleString()} steps
                  </p>
                </div>
                <button
                  onClick={() => { setActiveRun(null); loadHistory(); }}
                  className="shrink-0 text-xs text-slate-400 hover:text-emerald-500 transition-colors px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  ← {t('training.newTraining')}
                </button>
              </div>
              <TrainingMonitor
                runId={activeRun}
                onDone={() => { loadHistory(); }}
                onCancel={() => { setActiveRun(null); loadHistory(); }}
              />
            </div>
          ) : (
            <div className="card p-6">
              <div className="h-full flex flex-col items-center justify-center text-center gap-3 py-12">
                <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-full">
                  <Zap className="w-8 h-8 text-slate-400 dark:text-slate-500" />
                </div>
                <p className="text-slate-400 dark:text-slate-500 text-sm">{t('training.logEmpty')}</p>
              </div>
            </div>
          )}
        </div>

        {/* History */}
        {recentRuns.length > 0 && (
          <div className="card overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800">
              <h2 className="font-semibold text-slate-900 dark:text-white">{t('training.history')}</h2>
            </div>
            <div className="divide-y divide-slate-100 dark:divide-slate-800">
              {recentRuns.map(run => (
                <div key={run.id} className="px-5 py-3 flex items-center gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-slate-900 dark:text-white truncate">
                      &ldquo;{run.label ?? run.wakeWord}&rdquo;
                    </p>
                    <p className="text-xs text-slate-400">
                      {formatDate(run.createdAt)} · {run.samples} samples · {run.steps.toLocaleString()} steps
                      {run.fullMode && <span className="ml-1 text-emerald-500">· Full</span>}
                      {run.hasRealVoice && <span className="ml-1 text-blue-500">· Real voice</span>}
                      {run.platform === 'microWakeWord'
                        ? <span className="ml-1 text-violet-500">· ESP32</span>
                        : run.platform === 'both'
                          ? <span className="ml-1 text-emerald-500">· ESP32 + Wyoming</span>
                          : <span className="ml-1 text-sky-500">· Wyoming</span>
                      }
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <StatusBadge status={run.status} label={t(`status.${run.status}` as 'status.done')} />
                    <button
                      onClick={() => setActiveRun(run.id)}
                      className="text-xs text-slate-400 hover:text-emerald-500 transition-colors"
                    >
                      {run.status === 'running' ? 'Live' : 'Log'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
