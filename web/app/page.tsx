'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Package, Mic, Zap, Clock, ArrowRight, Pencil, Check, X } from 'lucide-react';
import { Header } from '@/components/header';
import { StatCard } from '@/components/stat-card';
import { StatusBadge } from '@/components/status-badge';
import { useI18n } from '@/lib/i18n';
import { formatDate } from '@/lib/utils';

interface Run {
  id: number;
  wakeWord: string;
  label: string | null;
  samples: number;
  steps: number;
  fullMode: boolean;
  hasRealVoice: boolean;
  status: string;
  createdAt: string;
  finishedAt: string | null;
}

interface Model {
  name: string;
  size: number;
  mtime: string;
}

interface Speaker {
  id: string;
  speaker: string;
  count: number;
}

function LabelEditor({ run, onSaved }: { run: Run; onSaved: () => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(run.label ?? run.wakeWord);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    if (!value.trim() || saving) return;
    setSaving(true);
    try {
      await fetch(`/api/train/${run.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: value.trim() }),
      });
      setEditing(false);
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  const cancel = () => {
    setValue(run.label ?? run.wakeWord);
    setEditing(false);
  };

  if (editing) {
    return (
      <span className="flex items-center gap-1">
        <input
          className="text-sm font-medium bg-slate-100 dark:bg-slate-800 border border-emerald-400 rounded px-1.5 py-0.5 text-slate-900 dark:text-white w-36 focus:outline-none"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') cancel(); }}
          autoFocus
        />
        <button onClick={save} disabled={saving} className="text-emerald-500 hover:text-emerald-600">
          <Check className="w-3.5 h-3.5" />
        </button>
        <button onClick={cancel} className="text-slate-400 hover:text-slate-600">
          <X className="w-3.5 h-3.5" />
        </button>
      </span>
    );
  }

  return (
    <span className="flex items-center gap-1 group/label">
      <span className="font-medium text-slate-900 dark:text-white truncate">
        &ldquo;{run.label ?? run.wakeWord}&rdquo;
      </span>
      <button
        onClick={() => setEditing(true)}
        className="opacity-0 group-hover/label:opacity-100 text-slate-400 hover:text-emerald-500 transition-all"
      >
        <Pencil className="w-3 h-3" />
      </button>
    </span>
  );
}

export default function DashboardPage() {
  const { t } = useI18n();
  const [runs, setRuns] = useState<Run[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);

  const load = () => {
    fetch('/api/train').then(r => r.json()).then(setRuns).catch(() => {});
    fetch('/api/models').then(r => r.json()).then(setModels).catch(() => {});
    fetch('/api/recordings').then(r => r.json()).then(setSpeakers).catch(() => {});
  };

  useEffect(() => { load(); }, []);

  const lastRun = runs[0];
  const totalRecordings = speakers.reduce((s, x) => s + x.count, 0);

  return (
    <div className="min-h-screen">
      <Header />
      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">{t('dashboard.title')}</h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">Wake Word Trainer for Home Assistant</p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <StatCard
            label={t('dashboard.modelsCount')}
            value={models.length}
            icon={Package}
            accent={models.length > 0}
          />
          <StatCard
            label={t('dashboard.recordingsCount')}
            value={totalRecordings}
            icon={Mic}
            accent={totalRecordings > 0}
            sub={speakers.map(s => s.speaker).join(', ') || undefined}
          />
          <StatCard
            label={t('dashboard.lastTraining')}
            value={lastRun ? formatDate(lastRun.finishedAt ?? lastRun.createdAt) : t('dashboard.noTraining')}
            icon={Clock}
            sub={lastRun ? (lastRun.label ?? lastRun.wakeWord) : undefined}
          />
        </div>

        {/* Quick actions */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Link
            href="/training"
            className="card p-5 flex items-center gap-4 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors group"
          >
            <div className="p-3 bg-emerald-100 dark:bg-emerald-950 rounded-xl text-emerald-600 dark:text-emerald-400">
              <Zap className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-slate-900 dark:text-white">{t('dashboard.btnTrain')}</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">TTS + echte Stimmen → .onnx</p>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-emerald-500 transition-colors" />
          </Link>

          <Link
            href="/recordings"
            className="card p-5 flex items-center gap-4 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors group"
          >
            <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-xl text-slate-500 dark:text-slate-400">
              <Mic className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-slate-900 dark:text-white">{t('dashboard.btnRecord')}</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Familie aufnehmen, 20× pro Person</p>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-emerald-500 transition-colors" />
          </Link>
        </div>

        {/* Recent training runs */}
        <div className="card overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800">
            <h2 className="font-semibold text-slate-900 dark:text-white">{t('dashboard.recentRuns')}</h2>
          </div>
          {runs.length === 0 ? (
            <div className="px-5 py-10 text-center">
              <p className="text-slate-400 dark:text-slate-500">{t('dashboard.noRuns')}</p>
              <Link href="/training" className="mt-3 inline-block text-sm text-emerald-600 dark:text-emerald-400 hover:underline">
                {t('dashboard.startFirst')}
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-slate-100 dark:divide-slate-800">
              {runs.slice(0, 8).map(run => (
                <div key={run.id} className="px-5 py-3 flex items-center gap-4 group">
                  <div className="flex-1 min-w-0">
                    <LabelEditor run={run} onSaved={load} />
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-0.5">
                      <span className="text-xs text-slate-400">{formatDate(run.createdAt)}</span>
                      <span className="text-xs text-slate-300 dark:text-slate-600">·</span>
                      <span className="text-xs text-slate-500 dark:text-slate-400">{run.samples} samples</span>
                      <span className="text-xs text-slate-300 dark:text-slate-600">·</span>
                      <span className="text-xs text-slate-500 dark:text-slate-400">{run.steps.toLocaleString()} steps</span>
                      {run.fullMode && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400 font-medium">Full</span>
                      )}
                      {run.hasRealVoice && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-400 font-medium">Real voice</span>
                      )}
                    </div>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge status={run.status} label={t(`status.${run.status}` as 'status.done')} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
