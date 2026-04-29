import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type { GenerationSummary } from '../../api/training';

interface GenerationComparisonChartProps {
  generations: GenerationSummary[];
}

interface ComparisonPoint {
  label: string;
  reward?: number;
  sharpe?: number;
  returnPercent?: number;
}

const toNumber = (value: number | string | boolean | null | undefined): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

/** Visual comparison for selected generations. */
export default function GenerationComparisonChart({
  generations,
}: GenerationComparisonChartProps): JSX.Element | null {
  if (generations.length < 2) {
    return null;
  }

  const data: ComparisonPoint[] = [...generations]
    .sort((left, right) => left.generation_number - right.generation_number)
    .map((generation) => ({
      label: `Gen ${generation.generation_number}`,
      reward: generation.final_reward ?? undefined,
      sharpe: toNumber(generation.metrics.sharpe_ratio),
      returnPercent:
        toNumber(generation.metrics.total_return) === undefined
          ? undefined
          : Number(toNumber(generation.metrics.total_return)) * 100,
    }));

  return (
    <section className="surface-panel p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold text-ink">Generation comparison</h3>
        <span className="status-pill bg-blue-50 text-action">{generations.length} selected</span>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
            <XAxis dataKey="label" tick={{ fontSize: 12 }} stroke="#78716c" />
            <YAxis yAxisId="reward" tick={{ fontSize: 12 }} stroke="#2563eb" width={56} />
            <YAxis
              yAxisId="ratio"
              orientation="right"
              tick={{ fontSize: 12 }}
              stroke="#1f7a4d"
              width={56}
            />
            <Tooltip formatter={(value: number | string) => value} />
            <Legend />
            <Bar
              yAxisId="reward"
              dataKey="reward"
              name="Final reward"
              fill="#2563eb"
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            />
            <Line
              yAxisId="ratio"
              type="monotone"
              dataKey="sharpe"
              name="Sharpe"
              stroke="#1f7a4d"
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              yAxisId="ratio"
              type="monotone"
              dataKey="returnPercent"
              name="Return %"
              stroke="#b7791f"
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
