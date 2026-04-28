import { Activity, CircleSlash2 } from 'lucide-react';

/** Route page for training job monitoring. */
export default function Training(): JSX.Element {
  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <div>
        <h2 className="text-2xl font-semibold text-ink">Training</h2>
        <p className="mt-1 text-sm text-stone-600">Queued and active model training jobs.</p>
      </div>

      <section className="surface-panel p-5">
        <div className="flex items-center justify-between gap-4 border-b border-stone-200 pb-4">
          <div className="flex items-center gap-3">
            <Activity className="text-success" size={22} aria-hidden="true" />
            <h3 className="font-semibold text-ink">Active job</h3>
          </div>
          <span className="status-pill bg-stone-100 text-stone-600">Idle</span>
        </div>
        <div className="flex min-h-56 flex-col items-center justify-center text-center text-stone-500">
          <CircleSlash2 size={30} aria-hidden="true" />
          <p className="mt-3 text-sm">No active training job.</p>
        </div>
      </section>
    </div>
  );
}