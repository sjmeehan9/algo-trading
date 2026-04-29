import { format } from 'date-fns';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type { EquityPoint } from '../../api/backtesting';

interface EquityCurveChartProps {
  data: EquityPoint[];
  initialCapital: number;
}

interface EquityChartPoint {
  timestamp: string;
  dateLabel: string;
  portfolioValue: number;
  cash: number;
  positionValue: number;
}

const formatCurrency = (value: number): string =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value);

/** Portfolio value and exposure chart for a backtest equity curve. */
export default function EquityCurveChart({
  data,
  initialCapital,
}: EquityCurveChartProps): JSX.Element {
  const chartData: EquityChartPoint[] = data.map((point) => ({
    timestamp: point.timestamp,
    dateLabel: format(new Date(point.timestamp), 'MMM d'),
    portfolioValue: point.portfolio_value,
    cash: point.cash,
    positionValue: point.position_value,
  }));

  return (
    <section className="surface-panel p-4">
      <h3 className="mb-4 font-semibold text-ink">Equity curve</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 18, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
            <XAxis dataKey="dateLabel" tick={{ fontSize: 12 }} stroke="#78716c" />
            <YAxis
              tick={{ fontSize: 12 }}
              stroke="#78716c"
              tickFormatter={(value: number) => `$${Math.round(value / 1000)}k`}
              width={64}
            />
            <Tooltip
              formatter={(value: number, name: string) => [
                formatCurrency(value),
                name === 'portfolioValue'
                  ? 'Portfolio value'
                  : name === 'positionValue'
                    ? 'Position value'
                    : 'Cash',
              ]}
              labelFormatter={(_, payload) => {
                const point = payload?.[0]?.payload as EquityChartPoint | undefined;
                return point ? format(new Date(point.timestamp), 'PPpp') : '';
              }}
            />
            <ReferenceLine y={initialCapital} stroke="#78716c" strokeDasharray="4 4" />
            <Area
              type="monotone"
              dataKey="positionValue"
              name="Position value"
              stroke="#b7791f"
              fill="#fef3c7"
              fillOpacity={0.45}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="portfolioValue"
              name="Portfolio value"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="cash"
              name="Cash"
              stroke="#1f7a4d"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
