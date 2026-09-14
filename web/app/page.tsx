'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Package, Mic, Zap, Clock, ArrowRight, Pencil, Check, X, FileArchive, Cpu, Server, AlertTriangle } from 'lucide-react';
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

interface RunInfo { id: number; status: string; createdAt: string; finishedAt: string | null }
interface Provenance { producedBy: RunInfo | null; latestRun: RunInfo | null; stale: boolean; staleReason: string | null }
interface Bundle { size: number; mtime: string; md5: string; provenance: Provenance }
interface Model {
  wakeWord: string;
  esp32: Bundle | null;
  wyoming: Bundle | null;
}

function downloadZip(wakeWord: string, platform: 'esp32' | 'wyoming', stale: boolean) {
  if (stale && !confirm('Dieses Modell ist laut Trainings-DB NICHT das neueste. Trotzdem herunterladen?')) return;
  const a = document.createElement('a');
  a.href = `/api/models/download?word=${encodeURIComponent(wakeWord)}&platform=${platform}${stale ? '&force=1' : ''}`;
  a.download = `${wakeWord}_${platform}.zip`;
  a.click();
}

function DownloadButton({ model, platform }: { model: Model; platform: 'esp32' | 'wyoming' }) {
  const b = platform === 'esp32' ? model.esp32 : model.wyoming;
  if (!b) return null;
  const Icon = platform === 'esp32' ? Cpu : Server;
  const stale = b.provenance.stale;
  const run = b.provenance.producedBy;
  return (
    <button
      onClick={() => downloadZip(model.wakeWord, platform, stale)}
      className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors text-left ${stale
        ? 'bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900'
        : platform === 'esp32'
          ? 'bg-violet-100 dark:bg-violet-950 text-violet-700 dark:text-violet-300 hover:bg-violet-200 dark:hover:bg-violet-900'
          : 'bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300 hover:bg-sky-200 dark:hover:bg-sky-900'}`}
      title={stale ? 'Veraltet – neuerer Trainingslauf vorhanden' : `md5 ${b.md5}`}
    >
      {stale ? <AlertTriangle className="w-4 h-4 shrink-0" /> : <Icon className="w-4 h-4 shrink-0" />}
      <span className="flex-1">
        <span className="block">{platform === 'esp32' ? 'ESP32 ZIP (.tflite + manifest)' : 'Wyoming ZIP (.onnx + .data)'}</span>
        <span className="block text-xs opacity-70 font-mono">
          {run ? `Run #${run.id} · ` : ''}{formatDate(b.mtime)}{stale ? ' · VERALTET' : ''}
        </span>
      </span>
      <FileArchive className="w-4 h-4 shrink-0 opacity-70" />
    </button>
  );
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

        {/* Newest model download */}
        {models[0] && (models[0].esp32 || models[0].wyoming) && (
          <div className="card p-5 flex flex-col sm:flex-row sm:items-center gap-4">
            <div className="p-3 bg-violet-100 dark:bg-violet-950 rounded-xl text-violet-600 dark:text-violet-400 self-start">
              <Package className="w-6 h-6" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-slate-900 dark:text-white">Neuestes Modell herunterladen</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                “{models[0].wakeWord.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}” ·
                Herkunft wird gegen die Trainings-DB geprüft, MD5 liegt im ZIP (BUILD_INFO.txt)
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-2">
              <DownloadButton model={models[0]} platform="esp32" />
              <DownloadButton model={models[0]} platform="wyoming" />
            </div>
          </div>
        )}

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
