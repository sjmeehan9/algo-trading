import clsx from 'clsx';
import { Activity, BarChart3, BrainCircuit, Home, Radio, Rocket, X } from 'lucide-react';
import { NavLink } from 'react-router-dom';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const navItems = [
  { path: '/', label: 'Dashboard', icon: Home },
  { path: '/models', label: 'Models', icon: BrainCircuit },
  { path: '/training', label: 'Training', icon: Activity },
  { path: '/backtesting', label: 'Backtesting', icon: BarChart3 },
  { path: '/deployment', label: 'Deployment', icon: Rocket },
  { path: '/trading', label: 'Trading', icon: Radio },
] as const;

/** Responsive application navigation sidebar. */
export default function Sidebar({ isOpen, onClose }: SidebarProps): JSX.Element {
  return (
    <>
      <div
        className={clsx(
          'fixed inset-0 z-30 bg-black/30 transition-opacity lg:hidden',
          isOpen ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
        aria-hidden="true"
        onClick={onClose}
      />
      <aside
        className={clsx(
          'fixed inset-y-0 left-0 z-40 flex w-72 flex-col bg-shell text-white transition-transform lg:static lg:z-auto lg:w-64 lg:translate-x-0',
          isOpen ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-white/10 px-5">
          <NavLink to="/" className="flex items-center gap-3" onClick={onClose}>
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-white text-shell">
              <BrainCircuit size={20} aria-hidden="true" />
            </span>
            <span className="text-base font-semibold tracking-normal">Algo Trading</span>
          </NavLink>
          <button
            className="icon-button border-white/10 bg-transparent text-white hover:bg-white/10 lg:hidden"
            type="button"
            aria-label="Close navigation"
            title="Close navigation"
            onClick={onClose}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <nav className="flex-1 px-3 py-4" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;

            return (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                onClick={onClose}
                className={({ isActive }) =>
                  clsx(
                    'mb-1 flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium text-stone-300 transition hover:bg-white/10 hover:text-white',
                    isActive && 'bg-white text-shell hover:bg-white hover:text-shell',
                  )
                }
              >
                <Icon size={18} aria-hidden="true" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </aside>
    </>
  );
}