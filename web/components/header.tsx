'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { useTheme } from 'next-themes';
import { useI18n } from '@/lib/i18n';
import { Mic2, LayoutDashboard, Zap, Mic, Package, Sun, Moon, Globe, Settings, X, Check } from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/', icon: LayoutDashboard, key: 'nav.dashboard' as const },
  { href: '/training', icon: Zap, key: 'nav.training' as const },
  { href: '/recordings', icon: Mic, key: 'nav.recordings' as const },
  { href: '/models', icon: Package, key: 'nav.models' as const },
];

const TTS_LANG_OPTIONS = [
  { value: 'de', flag: '🇩🇪', label: 'Deutsch', hint: 'de_DE Stimmen (Anna, Eddy, Flo …)' },
  { value: 'en', flag: '🇬🇧', label: 'English', hint: 'en_US/en_GB voices (Samantha, Daniel …)' },
] as const;

function SettingsModal({ onClose }: { onClose: () => void }) {
  const [haUrl, setHaUrl] = useState('');
  const [ttsLang, setTtsLang] = useState<'de' | 'en'>('de');
  const [saved, setSaved] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch('/api/settings').then(r => r.json()).then(s => {
      setHaUrl(s.haUrl ?? '');
      setTtsLang(s.ttsLang ?? 'de');
      setTimeout(() => inputRef.current?.select(), 50);
    });
  }, []);

  const save = async () => {
    await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ haUrl: haUrl.replace(/\/$/, ''), ttsLang }),
    });
    setSaved(true);
    setTimeout(onClose, 800);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24 px-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative w-full max-w-md bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 p-6 space-y-5">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-slate-900 dark:text-white flex items-center gap-2">
            <Settings className="w-4 h-4 text-emerald-500" />
            Einstellungen
          </h2>
          <button onClick={onClose} className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* TTS Language */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
            TTS-Sprache für Trainingssamples
          </label>
          <div className="grid grid-cols-2 gap-2">
            {TTS_LANG_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => { setTtsLang(opt.value); setSaved(false); }}
                className={cn(
                  'flex flex-col items-start gap-0.5 px-3 py-2.5 rounded-xl border text-left transition-colors',
                  ttsLang === opt.value
                    ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/40'
                    : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                )}
              >
                <span className="text-base">{opt.flag} <span className="text-sm font-semibold text-slate-900 dark:text-white">{opt.label}</span></span>
                <span className="text-xs text-slate-400">{opt.hint}</span>
              </button>
            ))}
          </div>
          <p className="text-xs text-amber-600 dark:text-amber-400 flex items-start gap-1">
            <span>⚠</span>
            <span>Muss zur Sprache des Wake Words passen — falsche Sprache verschlechtert das Modell erheblich.</span>
          </p>
        </div>

        {/* HA URL */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
            Home Assistant URL
          </label>
          <input
            ref={inputRef}
            value={haUrl}
            onChange={e => { setHaUrl(e.target.value); setSaved(false); }}
            onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') onClose(); }}
            placeholder="http://homeassistant.local:8123"
            className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
          <p className="text-xs text-slate-400">
            Wird automatisch in Manifest-URLs eingesetzt — einmal eintragen, nie wieder anpassen.
          </p>
        </div>

        <button
          onClick={save}
          className={cn(
            'w-full flex items-center justify-center gap-2 py-2.5 rounded-xl font-medium text-sm transition-colors',
            saved
              ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300'
              : 'bg-emerald-500 hover:bg-emerald-600 text-white'
          )}
        >
          {saved ? <><Check className="w-4 h-4" /> Gespeichert</> : 'Speichern'}
        </button>
      </div>
    </div>
  );
}

export function Header() {
  const { t, locale, setLocale } = useI18n();
  const { resolvedTheme, setTheme } = useTheme();
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <>
    <header className="sticky top-0 z-40 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md">
      <div className="max-w-5xl mx-auto px-4 h-14 flex items-center gap-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 font-bold text-slate-900 dark:text-white shrink-0">
          <Mic2 className="w-5 h-5 text-emerald-500" />
          <span className="hidden sm:block">Speaky</span>
        </Link>

        {/* Nav */}
        <nav className="flex items-center gap-1 flex-1">
          {navItems.map(({ href, icon: Icon, key }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                pathname === href
                  ? 'bg-emerald-50 dark:bg-emerald-950 text-emerald-600 dark:text-emerald-400'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white'
              )}
            >
              <Icon className="w-4 h-4" />
              <span className="hidden md:block">{t(key)}</span>
            </Link>
          ))}
        </nav>

        {/* Controls */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setSettingsOpen(true)}
            className="p-1.5 rounded-lg text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            title="Einstellungen"
          >
            <Settings className="w-4 h-4" />
          </button>
          <button
            onClick={() => setLocale(locale === 'de' ? 'en' : 'de')}
            className="flex items-center gap-1 px-2 py-1.5 rounded-lg text-sm text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            title="Switch language"
          >
            <Globe className="w-4 h-4" />
            <span className="font-medium">{locale.toUpperCase()}</span>
          </button>

          <button
            onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
            className="p-1.5 rounded-lg text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            title="Toggle theme"
          >
            {mounted && (resolvedTheme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />)}
          </button>
        </div>
      </div>
    </header>
    {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
  </>
  );
}
