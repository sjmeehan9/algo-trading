import { Trophy, X } from 'lucide-react';

import type {
  BacktestComparison as BacktestComparisonData,
  BacktestResult,
} from '../../api/backtesting';

interface BacktestComparisonProps {
  comparison: BacktestComparisonData;
  onClear: () => void;
}

const metricLabels: Record<string, string> = {
  total_return: 'Total return',
  total_return_dollars: 'Total P&L',
  annualized_return: 'Annualized',
  sharpe_ratio: 'Sharpe',
  sortino_ratio: 'Sortino',
  max_drawdown: 'Max drawdown',
  volatility: 'Volatility',
  win_rate: 'Win rate',
  profit_factor: 'Profit factor',
  total_trades: 'Trades',
};

const percentMetrics = new Set([
  'total_return',
  'annualized_return',
  'max_drawdown',
  'volatility',
  'win_rate',
]);
const currencyMetrics = new Set(['total_return_dollars']);

const formatMetricValue = (metricName: string, value: number | null): string => {
  if (value === null) {
    return '-';
  }
  if (currencyMetrics.has(metricName)) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(value);
  }
  if (percentMetrics.has(metricName)) {
    return `${(value * 100).toFixed(2)}%`;
  }
  if (metricName === 'total_trades') {
    return Math.round(value).toLocaleString();
  }
  return value.toFixed(2);
};

const labelForResult = (result: BacktestResult): string => {
  const symbols = result.request.symbols?.join(', ') || 'configured symbols';
  return `${symbols} ${result.request.start_date} to ${result.request.end_date}`;
};

/** Side-by-side comparison table for selected backtest results. */
export default function BacktestComparison({
  comparison,
  onClear,
}: BacktestComparisonProps): JSX.Element {
  return (
    <section className="surface-panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 px-5 py-4">
        <div>
          <h3 className="font-semibold text-ink">Backtest comparison</h3>
          <p className="mt-1 text-sm text-stone-600">
            Compare completed evaluations across core metrics.
          </p>
        </div>
        <button type="button" className="secondary-button" onClick={onClear}>
          <X size={16} aria-hidden="true" />
          Clear
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-stone-200 text-sm">
          <thead className="bg-stone-50 text-left text-xs uppercase text-stone-500">
            <tr>
              <th className="px-4 py-3 font-semibold">Metric</th>
              {comparison.backtests.map((result) => (
                <th key={result.backtest_id} className="min-w-48 px-4 py-3 font-semibold">
                  <div>{result.backtest_id}</div>
                  <div className="mt-1 normal-case text-stone-500">{labelForResult(result)}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-200 bg-white">
            {Object.entries(comparison.metric_comparison).map(([metricName, values]) => (
              <tr key={metricName}>
                <td className="px-4 py-3 font-medium text-ink">
                  {metricLabels[metricName] ?? metricName}
                </td>
                {values.map((value, index) => {
                  const result = comparison.backtests[index];
                  const isBest = result
                    ? comparison.best_by_metric[metricName] === result.backtest_id
                    : false;
                  return (
                    <td
                      key={`${metricName}-${result?.backtest_id ?? index}`}
                      className="px-4 py-3 text-stone-700"
                    >
                      <span
                        className={
                          isBest ? 'inline-flex items-center gap-1 font-semibold text-success' : ''
                        }
                      >
                        {isBest && <Trophy size={14} aria-hidden="true" />}
                        {formatMetricValue(metricName, value)}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
