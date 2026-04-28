import { Activity, BrainCircuit } from 'lucide-react';
import { useParams } from 'react-router-dom';

/** Route page for model configuration workflows. */
export default function ModelConfig(): JSX.Element {
  const { modelId } = useParams();
  const modeLabel = modelId ? 'Edit model' : 'New model';

  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <div>
        <div>
          <h2 className="text-2xl font-semibold text-ink">{modeLabel}</h2>
          <p className="mt-1 text-sm text-stone-600">Configuration workspace.</p>
        </div>
      </div>

      <section className="grid gap-4 md:grid-cols-2">
        <article className="surface-panel p-5">
          <BrainCircuit className="text-action" size={24} aria-hidden="true" />
          <h3 className="mt-4 font-semibold text-ink">Core RL model</h3>
          <dl className="mt-4 space-y-2 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-stone-500">Deployment</dt>
              <dd className="font-medium text-ink">Live trading eligible</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-stone-500">Inputs</dt>
              <dd className="font-medium text-ink">Market, strategies, signals</dd>
            </div>
          </dl>
        </article>

        <article className="surface-panel p-5">
          <Activity className="text-success" size={24} aria-hidden="true" />
          <h3 className="mt-4 font-semibold text-ink">Supporting model</h3>
          <dl className="mt-4 space-y-2 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-stone-500">Output</dt>
              <dd className="font-medium text-ink">Signal stream</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-stone-500">Inputs</dt>
              <dd className="font-medium text-ink">Market, news, indicators</dd>
            </div>
          </dl>
        </article>
      </section>
    </div>
  );
}