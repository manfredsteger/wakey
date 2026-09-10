'use client';

import { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import {
  Package, Trash2, FileAudio, Server, Cpu,
  Plus, X, Pencil, FolderOpen, Folder,
  ChevronDown, ChevronUp, FileArchive, GripVertical,
} from 'lucide-react';
import { Header } from '@/components/header';
import { useI18n } from '@/lib/i18n';
import { formatBytes, formatDate } from '@/lib/utils';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ModelFamily {
  wakeWord: string;
  esp32: { tflite: string; manifest: string | null; size: number; mtime: string } | null;
  wyoming: { onnx: string; data: string | null; size: number; mtime: string } | null;
}

interface Group {
  id: string;
  name: string;
  models: string[];
  notes: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function displayName(wakeWord: string) {
  return wakeWord.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function downloadZip(wakeWord: string, platform: 'esp32' | 'wyoming') {
  const a = document.createElement('a');
  a.href = `/api/models/download?word=${encodeURIComponent(wakeWord)}&platform=${platform}`;
  a.download = `${wakeWord}_${platform}.zip`;
  a.click();
}

async function patchGroup(id: string, patch: Partial<Group>) {
  await fetch('/api/models/groups', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, ...patch }),
  });
}

// ── GroupCard ─────────────────────────────────────────────────────────────────

function GroupCard({
  group, families, onUpdate, onDelete, dragOverId, setDragOverId,
}: {
  group: Group;
  families: ModelFamily[];
  onUpdate: () => void;
  onDelete: (id: string) => void;
  dragOverId: string | null;
  setDragOverId: (id: string | null) => void;
}) {
  const [open, setOpen] = useState(true);
  const [editing, setEditing] = useState(false);
  const [nameVal, setNameVal] = useState(group.name);
  const [addOpen, setAddOpen] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const isDragOver = dragOverId === group.id;

  const assigned = families.filter(f => group.models.includes(f.wakeWord));
  const available = families.filter(f => !group.models.includes(f.wakeWord));

  // Close dropdown on outside click
  useEffect(() => {
    if (!addOpen) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setAddOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [addOpen]);

  const saveName = async () => {
    if (!nameVal.trim()) { setEditing(false); return; }
    await patchGroup(group.id, { name: nameVal.trim() });
    setEditing(false);
    onUpdate();
  };

  const addModel = async (wakeWord: string) => {
    await patchGroup(group.id, { models: [...group.models, wakeWord] });
    setAddOpen(false);
    onUpdate();
  };

  const removeModel = async (wakeWord: string) => {
    await patchGroup(group.id, { models: group.models.filter(m => m !== wakeWord) });
    onUpdate();
  };

  // Drag-and-drop handlers (this card is a drop target)
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    const word = e.dataTransfer.types.includes('text/plain') ? 'check' : null;
    if (word) setDragOverId(group.id);
  };
  const onDragLeave = (e: React.DragEvent) => {
    // Only clear if leaving the card itself, not a child
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setDragOverId(null);
    }
  };
  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOverId(null);
    const wakeWord = e.dataTransfer.getData('text/plain');
    if (wakeWord && !group.models.includes(wakeWord)) {
      await addModel(wakeWord);
    }
  };

  return (
    <div
      className={`rounded-xl border-2 transition-colors ${
        isDragOver
          ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-950/30'
          : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900'
      }`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-200 dark:border-slate-700 rounded-t-xl bg-slate-50 dark:bg-slate-800/50">
        <button onClick={() => setOpen(o => !o)} className="shrink-0 text-emerald-500">
          {open ? <FolderOpen className="w-4 h-4" /> : <Folder className="w-4 h-4" />}
        </button>

        {editing ? (
          <input
            ref={nameRef}
            value={nameVal}
            autoFocus
            onChange={e => setNameVal(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') saveName();
              if (e.key === 'Escape') { setNameVal(group.name); setEditing(false); }
            }}
            onBlur={saveName}
            className="flex-1 font-semibold bg-transparent border-b border-emerald-500 focus:outline-none text-slate-900 dark:text-white"
          />
        ) : (
          <button
            onClick={() => setOpen(o => !o)}
            className="flex-1 text-left font-semibold text-slate-900 dark:text-white truncate"
          >
            {group.name}
          </button>
        )}

        <span className="text-xs text-slate-400 shrink-0">
          {group.models.length} Modell{group.models.length !== 1 ? 'e' : ''}
        </span>
        <button
          onClick={() => { setEditing(true); setTimeout(() => nameRef.current?.select(), 10); }}
          className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 shrink-0"
          title="Umbenennen"
        >
          <Pencil className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => { if (confirm(`Ordner "${group.name}" löschen?`)) onDelete(group.id); }}
          className="p-1 text-red-400 hover:text-red-600 shrink-0"
          title="Ordner löschen"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
        <button onClick={() => setOpen(o => !o)} className="p-1 text-slate-400 shrink-0">
          {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {open && (
        <div className="p-4 space-y-2">
          {isDragOver && (
            <div className="flex items-center justify-center gap-2 py-3 rounded-lg border-2 border-dashed border-emerald-400 text-emerald-600 dark:text-emerald-400 text-sm font-medium">
              <FolderOpen className="w-4 h-4" />
              Hier ablegen
            </div>
          )}

          {!isDragOver && assigned.length === 0 && (
            <p className="text-sm text-slate-400 dark:text-slate-500 py-3 text-center">
              Noch keine Modelle — Modell hierher ziehen oder „+ hinzufügen"
            </p>
          )}

          {assigned.map(f => (
            <div
              key={f.wakeWord}
              className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40"
            >
              <FileAudio className="w-4 h-4 text-emerald-500 shrink-0" />
              <span className="flex-1 font-medium text-sm text-slate-900 dark:text-white">
                {displayName(f.wakeWord)}
              </span>
              <div className="flex gap-1.5 shrink-0">
                {f.esp32 && (
                  <button
                    onClick={() => downloadZip(f.wakeWord, 'esp32')}
                    className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300 hover:bg-violet-200 dark:hover:bg-violet-900"
                  >
                    <Cpu className="w-3 h-3" /> ESP32 ZIP
                  </button>
                )}
                {f.wyoming && (
                  <button
                    onClick={() => downloadZip(f.wakeWord, 'wyoming')}
                    className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300 hover:bg-sky-200 dark:hover:bg-sky-900"
                  >
                    <Server className="w-3 h-3" /> Wyoming ZIP
                  </button>
                )}
              </div>
              <button
                onClick={() => removeModel(f.wakeWord)}
                className="p-1 text-slate-400 hover:text-red-500 shrink-0"
                title="Aus Ordner entfernen"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}

          {/* Add model button + dropdown */}
          <div ref={dropdownRef} className="relative">
            <button
              onClick={() => setAddOpen(o => !o)}
              disabled={available.length === 0}
              className="flex items-center gap-1.5 text-sm text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 disabled:opacity-40 disabled:cursor-not-allowed mt-1"
            >
              <Plus className="w-4 h-4" />
              {available.length === 0 ? 'Alle Modelle zugeordnet' : 'Modell hinzufügen'}
            </button>

            {addOpen && available.length > 0 && (
              <div className="absolute left-0 top-8 z-50 w-64 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl shadow-xl overflow-hidden">
                {available.map(f => (
                  <button
                    key={f.wakeWord}
                    onClick={() => addModel(f.wakeWord)}
                    className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-50 dark:hover:bg-slate-700 text-left"
                  >
                    <FileAudio className="w-4 h-4 text-emerald-500 shrink-0" />
                    <span className="flex-1 text-sm text-slate-900 dark:text-white">{displayName(f.wakeWord)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── ModelFamilyCard ───────────────────────────────────────────────────────────

function ModelFamilyCard({
  family, groups, onDelete, inGroup,
}: {
  family: ModelFamily;
  groups: Group[];
  onDelete: (wakeWord: string, platform: 'esp32' | 'wyoming' | 'all') => void;
  inGroup: boolean;
}) {
  const assignedGroups = groups.filter(g => g.models.includes(family.wakeWord));
  const hasBoth = family.esp32 && family.wyoming;

  const onDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData('text/plain', family.wakeWord);
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={`card p-4 space-y-3 cursor-grab active:cursor-grabbing select-none transition-opacity ${inGroup ? 'opacity-60' : ''}`}
    >
      {/* Title row */}
      <div className="flex items-start gap-3">
        <GripVertical className="w-4 h-4 text-slate-300 dark:text-slate-600 mt-0.5 shrink-0" />
        <div className="p-2 bg-emerald-100 dark:bg-emerald-950 rounded-xl shrink-0">
          <FileAudio className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-slate-900 dark:text-white">{displayName(family.wakeWord)}</p>
          <p className="text-xs text-slate-400 font-mono">{family.wakeWord}</p>
          {assignedGroups.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {assignedGroups.map(g => (
                <span
                  key={g.id}
                  className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 rounded-md bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                >
                  <Folder className="w-3 h-3" /> {g.name}
                </span>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => { if (confirm(`Alle Dateien für "${displayName(family.wakeWord)}" löschen?`)) onDelete(family.wakeWord, 'all'); }}
          className="btn-danger p-1.5 shrink-0"
          title="Alle löschen"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* ESP32 bundle */}
      {family.esp32 && (
        <div className="rounded-lg border border-violet-200 dark:border-violet-900 p-3 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="inline-flex items-center gap-1 text-xs font-medium px-1.5 py-0.5 rounded-md bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300">
              <Cpu className="w-3 h-3" /> ESP32
            </span>
            <span className="text-xs text-slate-400">
              {family.esp32.tflite}{family.esp32.manifest ? ' + manifest' : ''} · {formatBytes(family.esp32.size)} · {formatDate(family.esp32.mtime)}
            </span>
            {!hasBoth && (
              <button
                onClick={() => { if (confirm('ESP32-Modell löschen?')) onDelete(family.wakeWord, 'esp32'); }}
                className="ml-auto p-1 text-red-400 hover:text-red-600"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          <button
            onClick={() => downloadZip(family.wakeWord, 'esp32')}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-violet-100 dark:bg-violet-950 text-violet-700 dark:text-violet-300 hover:bg-violet-200 dark:hover:bg-violet-900 text-sm font-medium transition-colors"
          >
            <FileArchive className="w-4 h-4" />
            ZIP herunterladen (.tflite + manifest + README)
          </button>
        </div>
      )}

      {/* Wyoming bundle */}
      {family.wyoming && (
        <div className="rounded-lg border border-sky-200 dark:border-sky-900 p-3 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="inline-flex items-center gap-1 text-xs font-medium px-1.5 py-0.5 rounded-md bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300">
              <Server className="w-3 h-3" /> Wyoming
            </span>
            <span className="text-xs text-slate-400">
              {family.wyoming.onnx}{family.wyoming.data ? ' + .data' : ''} · {formatBytes(family.wyoming.size)} · {formatDate(family.wyoming.mtime)}
            </span>
            {!hasBoth && (
              <button
                onClick={() => { if (confirm('Wyoming-Modell löschen?')) onDelete(family.wakeWord, 'wyoming'); }}
                className="ml-auto p-1 text-red-400 hover:text-red-600"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          <button
            onClick={() => downloadZip(family.wakeWord, 'wyoming')}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300 hover:bg-sky-200 dark:hover:bg-sky-900 text-sm font-medium transition-colors"
          >
            <FileArchive className="w-4 h-4" />
            ZIP herunterladen (.onnx + .data + README)
          </button>
        </div>
      )}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function ModelsPage() {
  const { t } = useI18n();
  const [families, setFamilies] = useState<ModelFamily[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [loading, setLoading] = useState(true);
  const [newGroupName, setNewGroupName] = useState('');
  const [creatingGroup, setCreatingGroup] = useState(false);
  const [dragOverId, setDragOverId] = useState<string | null>(null);
  const newGroupRef = useRef<HTMLInputElement>(null);

  const loadAll = async () => {
    const [mRes, gRes] = await Promise.all([fetch('/api/models'), fetch('/api/models/groups')]);
    if (mRes.ok) setFamilies(await mRes.json());
    if (gRes.ok) setGroups(await gRes.json());
    setLoading(false);
  };

  useEffect(() => { loadAll(); }, []);

  const createGroup = async () => {
    if (!newGroupName.trim()) return;
    await fetch('/api/models/groups', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newGroupName.trim() }),
    });
    setNewGroupName('');
    setCreatingGroup(false);
    loadAll();
  };

  const deleteGroup = async (id: string) => {
    await fetch('/api/models/groups', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    loadAll();
  };

  const deleteModel = async (wakeWord: string, platform: 'esp32' | 'wyoming' | 'all') => {
    await fetch('/api/models', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wakeWord, platform }),
    });
    loadAll();
  };

  // Which wakeWords are assigned to at least one group
  const assignedWords = new Set(groups.flatMap(g => g.models));

  return (
    <div className="min-h-screen">
      <Header />
      <main className="max-w-5xl mx-auto px-4 py-8 space-y-10">

        {/* ── Ordner / Geräte ── */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <FolderOpen className="w-6 h-6 text-emerald-500" />
              Geräte / Ordner
            </h1>
            <button
              onClick={() => { setCreatingGroup(true); setTimeout(() => newGroupRef.current?.focus(), 50); }}
              className="btn-primary flex items-center gap-1.5 text-sm"
            >
              <Plus className="w-4 h-4" />
              Neuer Ordner
            </button>
          </div>

          {creatingGroup && (
            <div className="card p-4 flex gap-2 items-center">
              <FolderOpen className="w-4 h-4 text-emerald-500 shrink-0" />
              <input
                ref={newGroupRef}
                value={newGroupName}
                onChange={e => setNewGroupName(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') createGroup();
                  if (e.key === 'Escape') { setCreatingGroup(false); setNewGroupName(''); }
                }}
                placeholder="z.B. Wohnzimmer ReSpeaker"
                className="flex-1 px-3 py-1.5 text-sm rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
              <button onClick={createGroup} disabled={!newGroupName.trim()} className="btn-primary text-sm disabled:opacity-50">
                Erstellen
              </button>
              <button onClick={() => { setCreatingGroup(false); setNewGroupName(''); }} className="p-1.5 text-slate-400 hover:text-slate-600">
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {loading ? (
            <p className="text-slate-400">{t('common.loading')}</p>
          ) : groups.length === 0 && !creatingGroup ? (
            <div className="card p-10 text-center">
              <Folder className="w-10 h-10 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400 dark:text-slate-500">Noch keine Ordner — erstelle einen für jedes Gerät.</p>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">z.B. „Wohnzimmer ReSpeaker" oder „Küche ESP32"</p>
            </div>
          ) : (
            <div className="space-y-3">
              {groups.map(g => (
                <GroupCard
                  key={g.id}
                  group={g}
                  families={families}
                  onUpdate={loadAll}
                  onDelete={deleteGroup}
                  dragOverId={dragOverId}
                  setDragOverId={setDragOverId}
                />
              ))}
            </div>
          )}
        </section>

        {/* ── Alle Modelle ── */}
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Package className="w-5 h-5 text-emerald-500" />
              {t('models.title')}
            </h2>
            {families.length > 0 && (
              <span className="text-xs text-slate-400 mt-0.5">In Ordner ziehen zum Zuordnen</span>
            )}
          </div>

          {!loading && families.length === 0 && (
            <div className="card p-12 text-center">
              <Package className="w-8 h-8 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400 dark:text-slate-500">{t('models.noModels')}</p>
              <Link href="/training" className="mt-3 inline-block text-sm text-emerald-600 dark:text-emerald-400 hover:underline">
                {t('models.startTraining')}
              </Link>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {families.map(f => (
              <ModelFamilyCard
                key={f.wakeWord}
                family={f}
                groups={groups}
                onDelete={deleteModel}
                inGroup={assignedWords.has(f.wakeWord)}
              />
            ))}
          </div>
        </section>

      </main>
    </div>
  );
}
