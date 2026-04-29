import { useQuery } from '@tanstack/react-query';
import { AlertCircle, BarChart3, GitCompareArrows, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { backtestingApi, type BacktestRequest, type BacktestResult } from '../api/backtesting';
import { modelsApi, type ModelConfigResponse } from '../api/models';
import BacktestComparison from '../components/backtesting/BacktestComparison';
import BacktestForm, { type BacktestFormData } from '../components/backtesting/BacktestForm';
import BacktestResults from '../components/backtesting/BacktestResults';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useBacktest } from '../hooks/useBacktest';

const toMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to complete the backtesting request.';

const toModelNames = (models: ModelConfigResponse[]): Map<string, string> =>
  new Map(models.map((model) => [model.model_id, model.name]));

const formatPercent = (value: number | null | undefined): string =>
  typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '-';

const formatBacktestLabel = (result: BacktestResult, modelNames: Map<string, string>): string => {
  const modelName = modelNames.get(result.model_id) ?? result.model_id;
  const symbols = result.request.symbols?.join(', ') || 'configured symbols';
  return `${modelName} - ${symbols}`;
};

const selectedCompletedBacktests = (
  results: BacktestResult[],
  selectedIds: string[],
): BacktestResult[] =>
  selectedIds
    .map((backtestId) => results.find((result) => result.backtest_id === backtestId))
    .filter(
      (result): result is BacktestResult => result !== undefined && result.status === 'completed',
    );

/** Route page for backtesting configuration and result review. */
export default function Backtesting(): JSX.Element {
  const [selectedBacktestIds, setSelectedBacktestIds] = useState<string[]>([]);
  const {
    selectedResult,
    selectedTrades,
    comparison,
    exportCsv,
    clearComparison,
    runBacktest,
    compareBacktests,
    selectResult,
    isRunning,
    isLoadingTrades,
    isComparing,
    runError,
    tradeError,
    comparisonError,
  } = useBacktest();

  const modelsQuery = useQuery({
    queryKey: ['models', 'core_rl', 'backtesting-page'],
    queryFn: () => modelsApi.list({ modelType: 'core_rl', pageSize: 100 }),
  });
  const backtestsQuery = useQuery({
    queryKey: ['backtests'],
    queryFn: () => backtestingApi.list({ pageSize: 100 }),
    refetchInterval: 15_000,
  });

  const models = useMemo(() => modelsQuery.data?.items ?? [], [modelsQuery.data]);
  const modelNames = useMemo(() => toModelNames(models), [models]);
  const backtests = useMemo(() => backtestsQuery.data?.items ?? [], [backtestsQuery.data]);
  const comparableBacktests = useMemo(
    () => selectedCompletedBacktests(backtests, selectedBacktestIds),
    [backtests, selectedBacktestIds],
  );

  useEffect(() => {
    setSelectedBacktestIds((currentIds) =>
      currentIds.filter((backtestId) =>
        backtests.some((result) => result.backtest_id === backtestId),
      ),
    );
  }, [backtests]);

  const handleRunBacktest = (data: BacktestFormData): void => {
    const request: BacktestRequest = {
      model_id: data.model_id,
      generation_id: data.generation_id,
      start_date: data.start_date,
      end_date: data.end_date,
      initial_capital: data.initial_capital,
      symbols: data.symbols.length > 0 ? data.symbols : null,
      include_transaction_costs: data.include_transaction_costs,
      description: data.description.trim() || null,
    };
    runBacktest(request);
  };

  const toggleComparisonSelection = (backtestId: string): void => {
    setSelectedBacktestIds((currentIds) => {
      if (currentIds.includes(backtestId)) {
        return currentIds.filter((selectedId) => selectedId !== backtestId);
      }
      if (currentIds.length >= 5) {
        return currentIds;
      }
      return [...currentIds, backtestId];
    });
  };

  const handleCompare = (): void => {
    compareBacktests(comparableBacktests.map((result) => result.backtest_id));
  };

  const handleRefresh = (): void => {
    void Promise.all([modelsQuery.refetch(), backtestsQuery.refetch()]);
  };

  const requestError =
    runError ||
    tradeError ||
    comparisonError ||
    toMessage(modelsQuery.error || backtestsQuery.error);
  const hasRequestError = Boolean(
    runError || tradeError || comparisonError || modelsQuery.error || backtestsQuery.error,
  );

  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">Backtesting</h2>
          <p className="mt-1 text-sm text-stone-600">
            Run historical evaluations, inspect performance, and compare trained generations.
          </p>
        </div>
        <button type="button" className="secondary-button" onClick={handleRefresh}>
          <RefreshCw size={16} aria-hidden="true" />
          Refresh
        </button>
      </div>

      {hasRequestError && (
        <section
          className="surface-panel flex items-start gap-3 p-4 text-sm text-red-700"
          role="alert"
        >
          <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <div>
            <h3 className="font-semibold">Backtesting request failed</h3>
            <p className="mt-1">{requestError}</p>
          </div>
        </section>
      )}

      <section className="grid gap-5 xl:grid-cols-[minmax(340px,0.72fr)_minmax(0,1.28fr)]">
        <div className="space-y-5">
          <BacktestForm onSubmit={handleRunBacktest} isLoading={isRunning} />

          <section className="surface-panel overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 px-5 py-4">
              <div>
                <h3 className="font-semibold text-ink">Stored results</h3>
                <p className="mt-1 text-sm text-stone-600">Select results to open or compare.</p>
              </div>
              <span className="status-pill bg-stone-100 text-stone-600">
                {backtests.length} results
              </span>
            </div>

            {backtestsQuery.isLoading ? (
              <LoadingSpinner />
            ) : backtests.length === 0 ? (
              <div className="px-5 py-10 text-center text-sm text-stone-500">
                No backtests have been run yet.
              </div>
            ) : (
              <div className="divide-y divide-stone-200">
                {backtests.map((result) => {
                  const isSelected = selectedBacktestIds.includes(result.backtest_id);
                  const isComparable = result.status === 'completed';
                  return (
                    <div key={result.backtest_id} className="p-4 hover:bg-stone-50">
                      <div className="flex items-start gap-3">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          disabled={
                            !isComparable || (!isSelected && selectedBacktestIds.length >= 5)
                          }
                          onChange={() => toggleComparisonSelection(result.backtest_id)}
                          aria-label={`Compare ${result.backtest_id}`}
                          className="mt-1 h-4 w-4 rounded border-stone-300 text-action"
                        />
                        <div className="min-w-0 flex-1">
                          <button
                            type="button"
                            className="text-left font-semibold text-action hover:underline"
                            onClick={() => selectResult(result)}
                          >
                            {result.backtest_id}
                          </button>
                          <p className="mt-1 truncate text-sm text-stone-600">
                            {formatBacktestLabel(result, modelNames)}
                          </p>
                          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-stone-500">
                            <span className="status-pill bg-stone-100 text-stone-700">
                              {result.status}
                            </span>
                            <span>
                              {result.request.start_date} to {result.request.end_date}
                            </span>
                            <span>Return {formatPercent(result.metrics?.total_return)}</span>
                            <span>Sharpe {result.metrics?.sharpe_ratio?.toFixed(2) ?? '-'}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {selectedBacktestIds.length >= 2 && (
              <div className="border-t border-stone-200 bg-stone-50 px-5 py-4">
                <button
                  type="button"
                  className="primary-button w-full"
                  disabled={isComparing || comparableBacktests.length < 2}
                  onClick={handleCompare}
                >
                  <GitCompareArrows size={16} aria-hidden="true" />
                  {isComparing ? 'Comparing' : `Compare selected (${comparableBacktests.length})`}
                </button>
              </div>
            )}
          </section>
        </div>

        <div className="space-y-5">
          {comparison ? (
            <BacktestComparison comparison={comparison} onClear={clearComparison} />
          ) : null}

          {selectedResult || !comparison ? (
            <BacktestResults
              result={selectedResult}
              trades={selectedTrades}
              isLoadingTrades={isLoadingTrades}
              onExportTrades={() => {
                const filename = selectedResult
                  ? `${selectedResult.backtest_id}-trades.csv`
                  : 'backtest-trades.csv';
                exportCsv(selectedTrades, filename);
              }}
            />
          ) : (
            <section className="surface-panel flex min-h-96 items-center justify-center p-8 text-center text-stone-500">
              <div>
                <BarChart3 className="mx-auto text-stone-400" size={34} aria-hidden="true" />
                <p className="mt-3 text-sm">
                  Select a stored result to inspect the detailed trade log.
                </p>
              </div>
            </section>
          )}
        </div>
      </section>
    </div>
  );
}
