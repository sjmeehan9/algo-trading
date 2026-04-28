import { LockKeyhole, Rocket } from 'lucide-react';

/** Route page for deployment selection state. */
export default function Deployment(): JSX.Element {
  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <div>
        <h2 className="text-2xl font-semibold text-ink">Deployment</h2>
        <p className="mt-1 text-sm text-stone-600">Live-trading model selection.</p>
      </div>

      <section className="surface-panel p-5">
        <div className="flex items-center gap-3">
          <Rocket className="text-action" size={24} aria-hidden="true" />
          <h3 className="font-semibold text-ink">Selected model</h3>
        </div>
        <div className="mt-6 flex items-center gap-3 rounded-md border border-stone-200 bg-stone-50 p-4 text-sm text-stone-600">
          <LockKeyhole size={18} aria-hidden="true" />
          <span>No core RL model selected.</span>
        </div>
      </section>
    </div>
  );
}