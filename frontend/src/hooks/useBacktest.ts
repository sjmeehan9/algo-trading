import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';

import {
  backtestingApi,
  type BacktestComparison,
  type BacktestRequest,
  type BacktestResult,
  type TradeRecord,
} from '../api/backtesting';

export interface BacktestRunState {
  selectedResult: BacktestResult | null;
  selectedTrades: TradeRecord[];
  comparison: BacktestComparison | null;
  exportCsv: (trades: TradeRecord[], filename?: string) => void;
  clearComparison: () => void;
  runBacktest: (request: BacktestRequest) => void;
  compareBacktests: (backtestIds: string[]) => void;
  loadTrades: (backtestId: string) => void;
  selectResult: (result: BacktestResult) => void;
  isRunning: boolean;
  isLoadingTrades: boolean;
  isComparing: boolean;
  runError: string | null;
  tradeError: string | null;
  comparisonError: string | null;
}

const toMessage = (error: unknown, fallback: string): string =>
  error instanceof Error ? error.message : fallback;

const quoteCsvValue = (value: string | number | null | undefined): string => {
  if (value === null || value === undefined) {
    return '';
  }
  const stringValue = String(value);
  if (!/[",\n]/.test(stringValue)) {
    return stringValue;
  }
  return `"${stringValue.replace(/"/g, '""')}"`;
};

/** Convert trade records into a CSV string suitable for export. */
export const tradesToCsv = (trades: TradeRecord[]): string => {
  const columns: Array<keyof TradeRecord> = [
    'trade_id',
    'timestamp',
    'symbol',
    'action',
    'quantity',
    'price',
    'cost',
    'position_after',
    'portfolio_value',
    'signal_confidence',
  ];
  const header = columns.join(',');
  const rows = trades.map((trade) =>
    columns.map((column) => quoteCsvValue(trade[column] as string | number | null)).join(','),
  );
  return [header, ...rows].join('\n');
};

const downloadCsv = (csv: string, filename: string): void => {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

/** Coordinate backtest execution, trade loading, comparison, and CSV export. */
export function useBacktest(): BacktestRunState {
  const queryClient = useQueryClient();
  const [selectedResult, setSelectedResult] = useState<BacktestResult | null>(null);
  const [selectedTrades, setSelectedTrades] = useState<TradeRecord[]>([]);
  const [comparison, setComparison] = useState<BacktestComparison | null>(null);

  const runMutation = useMutation({
    mutationFn: (request: BacktestRequest) => backtestingApi.run(request),
    onSuccess: async (result) => {
      setSelectedResult(result);
      setComparison(null);
      if (result.status === 'completed') {
        const trades = await backtestingApi.getTrades(result.backtest_id);
        setSelectedTrades(trades);
      } else {
        setSelectedTrades([]);
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['backtests'] }),
        queryClient.invalidateQueries({ queryKey: ['backtests', result.model_id] }),
      ]);
    },
  });

  const tradesMutation = useMutation({
    mutationFn: (backtestId: string) => backtestingApi.getTrades(backtestId),
    onSuccess: (trades) => {
      setSelectedTrades(trades);
    },
  });

  const comparisonMutation = useMutation({
    mutationFn: (backtestIds: string[]) => backtestingApi.compare(backtestIds),
    onSuccess: (nextComparison) => {
      setComparison(nextComparison);
    },
  });

  const exportCsv = useCallback((trades: TradeRecord[], filename = 'backtest-trades.csv') => {
    downloadCsv(tradesToCsv(trades), filename);
  }, []);

  const clearComparison = useCallback(() => setComparison(null), []);
  const selectResult = useCallback(
    (result: BacktestResult) => {
      setSelectedResult(result);
      setComparison(null);
      tradesMutation.mutate(result.backtest_id);
    },
    [tradesMutation],
  );

  return useMemo(
    () => ({
      selectedResult,
      selectedTrades,
      comparison,
      exportCsv,
      clearComparison,
      runBacktest: (request: BacktestRequest) => runMutation.mutate(request),
      compareBacktests: (backtestIds: string[]) => comparisonMutation.mutate(backtestIds),
      loadTrades: (backtestId: string) => tradesMutation.mutate(backtestId),
      selectResult,
      isRunning: runMutation.isPending,
      isLoadingTrades: tradesMutation.isPending,
      isComparing: comparisonMutation.isPending,
      runError: runMutation.error ? toMessage(runMutation.error, 'Unable to run backtest.') : null,
      tradeError: tradesMutation.error
        ? toMessage(tradesMutation.error, 'Unable to load trades.')
        : null,
      comparisonError: comparisonMutation.error
        ? toMessage(comparisonMutation.error, 'Unable to compare backtests.')
        : null,
    }),
    [
      clearComparison,
      comparison,
      comparisonMutation,
      exportCsv,
      runMutation,
      selectedResult,
      selectedTrades,
      selectResult,
      tradesMutation,
    ],
  );
}
