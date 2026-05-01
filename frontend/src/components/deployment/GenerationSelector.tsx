import type { DeploymentGenerationSummary } from '../../api/deployment';

interface GenerationSelectorProps {
  generations: DeploymentGenerationSummary[];
  selectedGenerationId: string;
  onChange: (generationId: string) => void;
}

const formatMetric = (value: number | string | boolean | null | undefined): string => {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value.toFixed(3) : '-';
  }
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  return '-';
};

/** Generation picker with compact training/evaluation summaries. */
export default function GenerationSelector({
  generations,
  selectedGenerationId,
  onChange,
}: GenerationSelectorProps): JSX.Element {
  if (generations.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-stone-300 px-4 py-6 text-center text-sm text-stone-500">
        No trained generations are available for this model.
      </div>
    );
  }

  const selected = generations.find((generation) => generation.generation_id === selectedGenerationId);

  return (
    <div className="space-y-3">
      <label htmlFor="deployment-generation" className="block text-sm font-medium text-ink">
        Generation
      </label>
      <select
        id="deployment-generation"
        value={selectedGenerationId}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
      >
        {generations.map((generation) => (
          <option key={generation.generation_id} value={generation.generation_id}>
            Generation {generation.generation_number} / {generation.status}
          </option>
        ))}
      </select>
      {selected && (
        <dl className="grid gap-3 rounded-md border border-stone-200 bg-stone-50 p-3 text-sm sm:grid-cols-3">
          <div>
            <dt className="text-stone-500">Final reward</dt>
            <dd className="font-semibold text-ink">{formatMetric(selected.final_reward)}</dd>
          </div>
          <div>
            <dt className="text-stone-500">Sharpe</dt>
            <dd className="font-semibold text-ink">{formatMetric(selected.metrics.sharpe_ratio)}</dd>
          </div>
          <div>
            <dt className="text-stone-500">Return</dt>
            <dd className="font-semibold text-ink">{formatMetric(selected.metrics.total_return)}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}