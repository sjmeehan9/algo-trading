import { AlertCircle, CheckCircle2, Clock } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import type { BacktestResult, TradeRecord } from '../../api/backtesting';
import DrawdownChart from './DrawdownChart';
import EquityCurveChart from './EquityCurveChart';
import PerformanceMetrics from './PerformanceMetrics';
import TradeHistory from './TradeHistory';

interface BacktestResultsProps {
  result: BacktestResult | null;
  trades: TradeRecord[];
  isLoadingTrades?: boolean;
  onExportTrades: () => void;
}

const statusIcon = (status: BacktestResult['status']): JSX.Element => {
  if (status === 'completed') {
    return <CheckCircle2 size={18} aria-hidden="true" />;
  }
  if (status === 'failed') {
    return <AlertCircle size={18} aria-hidden="true" />;
  }
  return <Clock size={18} aria-hidden="true" />;
};

const statusClass = (status: BacktestResult['status']): string => {
  if (status === 'completed') {
    return 'bg-green-100 text-green-800';
  }
  if (status === 'failed') {
    return 'bg-red-100 text-red-800';
  }
  return 'bg-blue-100 text-action';
};

/** Full result display for one backtest run. */
export default function BacktestResults({
  result,
  trades,
  isLoadingTrades = false,
  onExportTrades,
}: BacktestResultsProps): JSX.Element {
  if (!result) {
    return (
      <section className="surface-panel flex min-h-96 items-center justify-center p-8 text-center text-stone-500">
        <div>
          <Clock className="mx-auto text-stone-400" size={34} aria-hidden="true" />
          <p className="mt-3 text-sm">
            Run a backtest or select a stored result to review evaluation details.
          </p>
        </div>
      </section>
    );
  }

  const createdAt = new Date(result.created_at);

  return (
    <div className="space-y-5">
      <section className="surface-panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-lg font-semibold text-ink">{result.backtest_id}</h3>
              <span className={`status-pill gap-1 ${statusClass(result.status)}`}>
                {statusIcon(result.status)}
                {result.status}
              </span>
            </div>
            <p className="mt-1 text-sm text-stone-600">
              Created {formatDistanceToNow(createdAt)} ago for generation {result.generation_id}
            </p>
            {result.request.description && (
              <p className="mt-2 text-sm text-stone-700">{result.request.description}</p>
            )}
          </div>
          <div className="grid gap-2 text-right text-sm text-stone-600 sm:grid-cols-2 sm:text-left">
            <div>
              <div className="font-semibold text-ink">Window</div>
              <div>
                {result.request.start_date} to {result.request.end_date}
              </div>
            </div>
            <div>
              <div className="font-semibold text-ink">Symbols</div>
              <div>{result.request.symbols?.join(', ') || 'Configured symbols'}</div>
            </div>
          </div>
        </div>
        {result.error_message && (
          <div
            className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
            role="alert"
          >
            {result.error_message}
          </div>
        )}
      </section>

      {result.metrics ? <PerformanceMetrics metrics={result.metrics} /> : null}

      {result.equity_curve.length > 0 ? (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
          <EquityCurveChart
            data={result.equity_curve}
            initialCapital={result.request.initial_capital}
          />
          <DrawdownChart data={result.equity_curve} />
        </div>
      ) : null}

      <TradeHistory trades={trades} isLoading={isLoadingTrades} onExport={onExportTrades} />
    </div>
  );
}
