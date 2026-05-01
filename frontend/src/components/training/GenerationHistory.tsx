import { ChevronDown, ChevronRight } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { Fragment, useMemo, useState } from 'react';

import type { GenerationSummary } from '../../api/training';

interface GenerationHistoryProps {
  generations: GenerationSummary[];
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
}

const MAX_COMPARISON_GENERATIONS = 5;

const formatDuration = (seconds: number): string => {
  const safeSeconds = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
};

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

const toComparableNumber = (value: number | string | boolean | null | undefined): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/** Generation table with comparison selection and expandable metric details. */
export default function GenerationHistory({
  generations,
  selectedIds,
  onSelectionChange,
}: GenerationHistoryProps): JSX.Element {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const orderedGenerations = useMemo(
    () => [...generations].sort((left, right) => right.generation_number - left.generation_number),
    [generations],
  );

  const toggleSelection = (generationId: string): void => {
    if (selectedIds.includes(generationId)) {
      onSelectionChange(selectedIds.filter((selectedId) => selectedId !== generationId));
      return;
    }
    if (selectedIds.length < MAX_COMPARISON_GENERATIONS) {
      onSelectionChange([...selectedIds, generationId]);
    }
  };

  if (orderedGenerations.length === 0) {
    return (
      <section className="surface-panel px-5 py-10 text-center text-sm text-stone-500">
        No generations found for this model.
      </section>
    );
  }

  return (
    <section className="surface-panel overflow-hidden">
      <div className="border-b border-stone-200 px-5 py-4">
        <h3 className="font-semibold text-ink">Generation history</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-stone-200 text-sm">
          <thead className="bg-stone-50 text-left text-xs uppercase text-stone-500">
            <tr>
              <th className="px-4 py-3 font-semibold">Compare</th>
              <th className="px-4 py-3 font-semibold">Generation</th>
              <th className="px-4 py-3 font-semibold">Trained</th>
              <th className="px-4 py-3 font-semibold">Duration</th>
              <th className="px-4 py-3 font-semibold">Final reward</th>
              <th className="px-4 py-3 font-semibold">Sharpe</th>
              <th className="px-4 py-3 font-semibold">Return</th>
              <th className="px-4 py-3 font-semibold">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-200 bg-white">
            {orderedGenerations.map((generation, index) => {
              const olderGeneration = orderedGenerations[index + 1];
              const currentReward = generation.final_reward ?? null;
              const olderReward = olderGeneration?.final_reward ?? null;
              const improved =
                typeof currentReward === 'number' &&
                typeof olderReward === 'number' &&
                currentReward > olderReward;
              const isSelected = selectedIds.includes(generation.generation_id);
              const isDisabled = !isSelected && selectedIds.length >= MAX_COMPARISON_GENERATIONS;
              const sharpe = toComparableNumber(generation.metrics.sharpe_ratio);
              const totalReturn = toComparableNumber(generation.metrics.total_return);
              const isExpanded = expandedId === generation.generation_id;
              const metricEntries = Object.entries(generation.metrics).slice(0, 8);

              return (
                <Fragment key={generation.generation_id}>
                  <tr className="hover:bg-stone-50">
                    <td className="px-4 py-3">
                      <label className="inline-flex items-center gap-2 text-xs text-stone-600">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          disabled={isDisabled}
                          onChange={() => toggleSelection(generation.generation_id)}
                          className="h-4 w-4 rounded border-stone-300 text-action"
                        />
                        Compare Generation {generation.generation_number}
                      </label>
                    </td>
                    <td className="px-4 py-3 font-medium text-ink">
                      Gen {generation.generation_number}
                    </td>
                    <td className="px-4 py-3 text-stone-600">
                      {formatDistanceToNow(new Date(generation.created_at))} ago
                    </td>
                    <td className="px-4 py-3 text-stone-700">
                      {formatDuration(generation.training_duration_seconds)}
                    </td>
                    <td className="px-4 py-3">
                      <span className={improved ? 'font-semibold text-success' : 'text-ink'}>
                        {currentReward === null ? '-' : currentReward.toFixed(3)}
                        {improved ? ' ↑' : ''}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-stone-700">
                      {sharpe === null ? '-' : sharpe.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-stone-700">
                      {totalReturn === null ? '-' : `${(totalReturn * 100).toFixed(2)}%`}
                    </td>
                    <td className="px-4 py-3">
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-sm font-medium text-action hover:bg-blue-50"
                        onClick={() => setExpandedId(isExpanded ? null : generation.generation_id)}
                      >
                        {isExpanded ? (
                          <ChevronDown size={14} aria-hidden="true" />
                        ) : (
                          <ChevronRight size={14} aria-hidden="true" />
                        )}
                        View
                      </button>
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr className="bg-stone-50">
                      <td colSpan={8} className="px-4 py-4">
                        {metricEntries.length > 0 ? (
                          <div className="flex flex-wrap gap-2">
                            {metricEntries.map(([metricName, value]) => (
                              <span
                                key={metricName}
                                className="rounded-md border border-stone-200 bg-white px-2.5 py-1 text-xs text-stone-700"
                              >
                                <span className="font-semibold text-ink">{metricName}</span>:{' '}
                                {formatMetric(value)}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-sm text-stone-500">No metrics captured.</span>
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
