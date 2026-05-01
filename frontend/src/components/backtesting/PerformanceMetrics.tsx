import { Activity, BarChart3, DollarSign, ShieldAlert } from 'lucide-react';

import type { PerformanceMetrics as PerformanceMetricsData } from '../../api/backtesting';

interface PerformanceMetricsProps {
  metrics: PerformanceMetricsData;
}

interface MetricItem {
  label: string;
  value: string;
  tone?: 'positive' | 'negative' | 'neutral';
}

interface MetricGroup {
  title: string;
  icon: typeof DollarSign;
  items: MetricItem[];
}

const formatPercent = (value: number): string => `${(value * 100).toFixed(2)}%`;

const formatCurrency = (value: number): string =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value);

const formatNumber = (value: number, digits = 2): string =>
  Number.isFinite(value) ? value.toFixed(digits) : '-';

const toneClass = (tone: MetricItem['tone']): string => {
  if (tone === 'positive') {
    return 'text-success';
  }
  if (tone === 'negative') {
    return 'text-red-700';
  }
  return 'text-ink';
};

/** Grouped performance metric cards for a completed backtest. */
export default function PerformanceMetrics({ metrics }: PerformanceMetricsProps): JSX.Element {
  const groups: MetricGroup[] = [
    {
      title: 'Returns',
      icon: DollarSign,
      items: [
        {
          label: 'Total return',
          value: formatPercent(metrics.total_return),
          tone: metrics.total_return >= 0 ? 'positive' : 'negative',
        },
        {
          label: 'Total P&L',
          value: formatCurrency(metrics.total_return_dollars),
          tone: metrics.total_return_dollars >= 0 ? 'positive' : 'negative',
        },
        { label: 'Annualized', value: formatPercent(metrics.annualized_return) },
      ],
    },
    {
      title: 'Risk',
      icon: ShieldAlert,
      items: [
        { label: 'Sharpe', value: formatNumber(metrics.sharpe_ratio) },
        { label: 'Sortino', value: formatNumber(metrics.sortino_ratio) },
        { label: 'Max drawdown', value: formatPercent(metrics.max_drawdown), tone: 'negative' },
        { label: 'Volatility', value: formatPercent(metrics.volatility) },
      ],
    },
    {
      title: 'Trades',
      icon: Activity,
      items: [
        { label: 'Total trades', value: metrics.total_trades.toLocaleString() },
        { label: 'Win rate', value: formatPercent(metrics.win_rate) },
        { label: 'Profit factor', value: formatNumber(metrics.profit_factor) },
        { label: 'Exposure', value: formatPercent(metrics.exposure_time) },
      ],
    },
    {
      title: 'Trade values',
      icon: BarChart3,
      items: [
        { label: 'Average win', value: formatCurrency(metrics.average_win), tone: 'positive' },
        { label: 'Average loss', value: formatCurrency(metrics.average_loss), tone: 'negative' },
        { label: 'Largest win', value: formatCurrency(metrics.largest_win), tone: 'positive' },
        { label: 'Largest loss', value: formatCurrency(metrics.largest_loss), tone: 'negative' },
      ],
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {groups.map((group) => {
        const Icon = group.icon;
        return (
          <section key={group.title} className="surface-panel p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-ink">
              <Icon size={16} aria-hidden="true" />
              {group.title}
            </div>
            <dl className="space-y-2">
              {group.items.map((item) => (
                <div key={item.label} className="flex items-center justify-between gap-3 text-sm">
                  <dt className="text-stone-600">{item.label}</dt>
                  <dd className={`font-semibold ${toneClass(item.tone)}`}>{item.value}</dd>
                </div>
              ))}
            </dl>
          </section>
        );
      })}
    </div>
  );
}
