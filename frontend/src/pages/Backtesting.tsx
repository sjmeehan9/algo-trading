import { BarChart3, CalendarRange } from 'lucide-react';

/** Route page for backtesting configuration and result review. */
export default function Backtesting(): JSX.Element {
  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <div>
        <h2 className="text-2xl font-semibold text-ink">Backtesting</h2>
        <p className="mt-1 text-sm text-stone-600">Historical evaluation results and comparisons.</p>
      </div>

      <section className="grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
        <div className="surface-panel p-5">
          <div className="flex items-center gap-3">
            <CalendarRange className="text-caution" size={22} aria-hidden="true" />
            <h3 className="font-semibold text-ink">Backtest setup</h3>
          </div>
          <div className="mt-5 grid gap-4 text-sm">
            <div className="flex items-center justify-between gap-4 rounded-md bg-stone-50 p-3">
              <span className="font-medium text-stone-700">Model</span>
              <span className="text-stone-500">None selected</span>
            </div>
            <div className="flex items-center justify-between gap-4 rounded-md bg-stone-50 p-3">
              <span className="font-medium text-stone-700">Generation</span>
              <span className="text-stone-500">None selected</span>
            </div>
          </div>
        </div>

        <div className="surface-panel flex min-h-80 items-center justify-center p-5 text-center text-stone-500">
          <div>
            <BarChart3 className="mx-auto" size={34} aria-hidden="true" />
            <p className="mt-3 text-sm">No backtest result selected.</p>
          </div>
        </div>
      </section>
    </div>
  );
}