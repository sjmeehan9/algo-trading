import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertCircle, BrainCircuit, RefreshCw, Sparkles } from 'lucide-react';
import { useMemo, useState } from 'react';

import { optimizerApi, type OptimizationResult } from '../../api/optimizer';
import LoadingSpinner from '../common/LoadingSpinner';
import SuggestionCard from './SuggestionCard';

interface Props {
  modelId: string | null;
}

const toMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to complete optimizer request.';

const resultTimestamp = (result: OptimizationResult): string =>
  new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(result.created_at));

/** Panel for analyzing and applying hyperparameter optimizer suggestions. */
export default function HyperparameterSuggestions({ modelId }: Props): JSX.Element {
  const queryClient = useQueryClient();
  const [localResult, setLocalResult] = useState<OptimizationResult | null>(null);

  const latestQuery = useQuery({
    queryKey: ['optimizer', 'latest', modelId],
    queryFn: () => optimizerApi.getLatest(modelId ?? ''),
    enabled: Boolean(modelId),
  });

  const activeResult = useMemo(
    () => localResult ?? latestQuery.data ?? null,
    [latestQuery.data, localResult],
  );

  const analyzeMutation = useMutation({
    mutationFn: () => {
      if (!modelId) {
        throw new Error('Select a model before requesting suggestions.');
      }
      return optimizerApi.analyze({
        model_id: modelId,
        max_generations: 5,
        include_backtest_metrics: true,
      });
    },
    onSuccess: async (result) => {
      setLocalResult(result);
      await queryClient.invalidateQueries({ queryKey: ['optimizer', 'latest', modelId] });
    },
  });

  const applyMutation = useMutation({
    mutationFn: (suggestionId: string) => {
      if (!modelId || !activeResult) {
        throw new Error('Run optimizer analysis before applying a suggestion.');
      }
      return optimizerApi.apply({
        model_id: modelId,
        result_id: activeResult.result_id,
        suggestion_id: suggestionId,
      });
    },
    onSuccess: async (response) => {
      setLocalResult(response.optimization_result);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['models'] }),
        queryClient.invalidateQueries({ queryKey: ['optimizer', 'latest', modelId] }),
      ]);
    },
  });

  const requestError =
    analyzeMutation.error || applyMutation.error || latestQuery.error
      ? toMessage(analyzeMutation.error || applyMutation.error || latestQuery.error)
      : null;

  return (
    <section className="surface-panel overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-stone-200 px-5 py-4">
        <div>
          <h3 className="font-semibold text-ink">Hyperparameter optimizer</h3>
          <p className="mt-1 text-sm text-stone-600">
            Analyze recent generations and apply the next configuration change.
          </p>
        </div>
        <button
          type="button"
          className="primary-button"
          disabled={!modelId || analyzeMutation.isPending}
          onClick={() => analyzeMutation.mutate()}
        >
          {analyzeMutation.isPending ? (
            <RefreshCw size={16} aria-hidden="true" />
          ) : (
            <Sparkles size={16} aria-hidden="true" />
          )}
          {analyzeMutation.isPending ? 'Analyzing' : 'Analyze'}
        </button>
      </div>

      {requestError ? (
        <div className="flex items-start gap-3 border-b border-stone-200 bg-red-50 px-5 py-4 text-sm text-red-700">
          <AlertCircle className="mt-0.5 shrink-0" size={17} aria-hidden="true" />
          <p>{requestError}</p>
        </div>
      ) : null}

      {latestQuery.isLoading ? (
        <div className="p-5">
          <LoadingSpinner />
        </div>
      ) : activeResult ? (
        <div className="space-y-4 p-5">
          <div className="flex flex-wrap items-center gap-2 text-xs text-stone-500">
            <span className="status-pill bg-stone-100 text-stone-700">{activeResult.source}</span>
            <span>{resultTimestamp(activeResult)}</span>
            <span>{activeResult.analyzed_generation_ids.length} generations</span>
          </div>
          <p className="text-sm text-stone-700">{activeResult.summary}</p>
          {activeResult.priority_changes.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {activeResult.priority_changes.map((change) => (
                <span key={change} className="status-pill bg-blue-50 text-blue-700">
                  {change}
                </span>
              ))}
            </div>
          ) : null}
          {activeResult.suggestions.length > 0 ? (
            <div className="space-y-3">
              {activeResult.suggestions.map((suggestion) => (
                <SuggestionCard
                  key={suggestion.suggestion_id}
                  suggestion={suggestion}
                  isApplying={applyMutation.variables === suggestion.suggestion_id}
                  onApply={() => applyMutation.mutate(suggestion.suggestion_id)}
                />
              ))}
            </div>
          ) : (
            <div className="rounded-md border border-dashed border-stone-300 px-4 py-8 text-center text-sm text-stone-500">
              No actionable suggestions were returned for this analysis.
            </div>
          )}
        </div>
      ) : (
        <div className="px-5 py-10 text-center text-sm text-stone-500">
          <BrainCircuit className="mx-auto mb-3 text-stone-400" size={28} aria-hidden="true" />
          Select a model and run analysis after training history is available.
        </div>
      )}
    </section>
  );
}
