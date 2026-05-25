import { Menu, RefreshCw } from 'lucide-react';
import { useMemo } from 'react';
import { useLocation } from 'react-router-dom';

interface HeaderProps {
  onMenuClick: () => void;
}

const pathTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/models': 'Models',
  '/training': 'Training',
  '/backtesting': 'Backtesting',
  '/deployment': 'Deployment',
  '/trading': 'Trading',
};

const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/** Top application bar with route context and backend target state. */
export default function Header({ onMenuClick }: HeaderProps): JSX.Element {
  const location = useLocation();
  const title = useMemo(() => pathTitles[location.pathname] || 'Model Workspace', [location.pathname]);

  return (
    <header className="flex h-16 items-center justify-between border-b border-stone-200 bg-white px-4 lg:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <button
          className="icon-button lg:hidden"
          type="button"
          aria-label="Open navigation"
          title="Open navigation"
          onClick={onMenuClick}
        >
          <Menu size={18} aria-hidden="true" />
        </button>
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wide text-stone-500">Model Builder</p>
          <h1 className="truncate text-xl font-semibold text-ink">{title}</h1>
        </div>
      </div>

      <div className="hidden items-center gap-3 text-sm text-stone-600 sm:flex">
        <span className="status-pill bg-emerald-50 text-success">API</span>
        <span className="max-w-64 truncate">{apiBaseUrl}</span>
        <button
          className="icon-button"
          type="button"
          aria-label="Refresh view"
          title="Refresh view"
          onClick={() => window.location.reload()}
        >
          <RefreshCw size={16} aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}