import { cn } from '@/lib/utils';

const styles: Record<string, string> = {
  pending: 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400',
  running: 'bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-400 animate-pulse',
  done: 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400',
  failed: 'bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-400',
  paused: 'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-400',
  cancelled: 'bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-400',
};

const dots: Record<string, string> = {
  pending: 'bg-slate-400',
  running: 'bg-blue-500',
  done: 'bg-emerald-500',
  failed: 'bg-red-500',
  paused: 'bg-amber-500',
  cancelled: 'bg-red-400',
};

interface StatusBadgeProps {
  status: string;
  label: string;
}

export function StatusBadge({ status, label }: StatusBadgeProps) {
  return (
    <span className={cn('badge gap-1.5', styles[status] ?? styles.pending)}>
      <span className={cn('w-1.5 h-1.5 rounded-full', dots[status] ?? dots.pending)} />
      {label}
    </span>
  );
}
