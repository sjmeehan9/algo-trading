import { format } from 'date-fns';
import {
  Area,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  AreaChart,
} from 'recharts';

import type { EquityPoint } from '../../api/backtesting';

interface DrawdownChartProps {
  data: EquityPoint[];
}

interface DrawdownPoint {
  timestamp: string;
  dateLabel: string;
  drawdownPercent: number;
}

/** Drawdown visualization for a backtest equity curve. */
export default function DrawdownChart({ data }: DrawdownChartProps): JSX.Element {
  const chartData: DrawdownPoint[] = data.map((point) => ({
    timestamp: point.timestamp,
    dateLabel: format(new Date(point.timestamp), 'MMM d'),
    drawdownPercent: point.drawdown * 100,
  }));

  return (
    <section className="surface-panel p-4">
      <h3 className="mb-4 font-semibold text-ink">Drawdown</h3>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 8, right: 18, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
            <XAxis dataKey="dateLabel" tick={{ fontSize: 12 }} stroke="#78716c" />
            <YAxis
              tick={{ fontSize: 12 }}
              stroke="#78716c"
              tickFormatter={(value: number) => `${value.toFixed(0)}%`}
              width={52}
            />
            <Tooltip
              formatter={(value: number) => [`${value.toFixed(2)}%`, 'Drawdown']}
              labelFormatter={(_, payload) => {
                const point = payload?.[0]?.payload as DrawdownPoint | undefined;
                return point ? format(new Date(point.timestamp), 'PPpp') : '';
              }}
            />
            <Area
              type="monotone"
              dataKey="drawdownPercent"
              stroke="#b91c1c"
              fill="#fee2e2"
              fillOpacity={0.75}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
