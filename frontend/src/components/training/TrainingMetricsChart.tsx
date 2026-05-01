import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type { TrainingMetricsHistoryPoint } from '../../hooks/useTrainingProgress';

interface TrainingMetricsChartProps {
  data: TrainingMetricsHistoryPoint[];
}

const formatNumber = (value: number | string): string => {
  if (typeof value === 'number') {
    return Math.abs(value) >= 1 ? value.toFixed(2) : value.toFixed(5);
  }
  return value;
};

/** Live reward and loss chart for an active training job. */
export default function TrainingMetricsChart({ data }: TrainingMetricsChartProps): JSX.Element {
  const hasReward = data.some((point) => point.reward !== undefined);
  const hasLoss = data.some((point) => point.loss !== undefined);

  if (!hasReward && !hasLoss) {
    return (
      <div className="flex h-full min-h-48 items-center justify-center rounded-md border border-dashed border-stone-300 text-sm text-stone-500">
        Waiting for reward or loss metrics.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
        <XAxis
          dataKey="timestep"
          tickFormatter={(value: number) => `${Math.round(value / 1000)}k`}
          stroke="#78716c"
          tick={{ fontSize: 12 }}
        />
        <YAxis yAxisId="reward" stroke="#2563eb" tick={{ fontSize: 12 }} width={56} allowDecimals />
        {hasLoss && (
          <YAxis
            yAxisId="loss"
            orientation="right"
            stroke="#b7791f"
            tick={{ fontSize: 12 }}
            width={56}
            allowDecimals
          />
        )}
        <Tooltip
          formatter={(value: number | string, name: string) => [
            formatNumber(value),
            name === 'reward' ? 'Reward' : 'Loss',
          ]}
          labelFormatter={(value: number) => `Timestep ${value.toLocaleString()}`}
        />
        <Legend />
        {hasReward && (
          <Line
            yAxisId="reward"
            type="monotone"
            dataKey="reward"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            name="Reward"
            connectNulls
            isAnimationActive={false}
          />
        )}
        {hasLoss && (
          <Line
            yAxisId="loss"
            type="monotone"
            dataKey="loss"
            stroke="#b7791f"
            strokeWidth={2}
            dot={false}
            name="Loss"
            connectNulls
            isAnimationActive={false}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
