import { ArrowDownUp, Download } from 'lucide-react';
import { format } from 'date-fns';
import { useMemo, useState } from 'react';

import type { TradeAction, TradeRecord } from '../../api/backtesting';

interface TradeHistoryProps {
  trades: TradeRecord[];
  onExport: () => void;
  isLoading?: boolean;
}

type TradeFilter = 'all' | Extract<TradeAction, 'BUY' | 'SELL'>;
type SortField = 'timestamp' | 'symbol' | 'action' | 'quantity' | 'price' | 'portfolio_value';
type SortDirection = 'asc' | 'desc';

const sortableHeaders: Array<{ field: SortField; label: string }> = [
  { field: 'timestamp', label: 'Time' },
  { field: 'symbol', label: 'Symbol' },
  { field: 'action', label: 'Action' },
  { field: 'quantity', label: 'Quantity' },
  { field: 'price', label: 'Price' },
  { field: 'portfolio_value', label: 'Portfolio' },
];

const compareValues = (
  left: string | number,
  right: string | number,
  direction: SortDirection,
): number => {
  const multiplier = direction === 'asc' ? 1 : -1;
  if (left < right) {
    return -1 * multiplier;
  }
  if (left > right) {
    return 1 * multiplier;
  }
  return 0;
};

const formatCurrency = (value: number): string =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

const actionClass = (action: TradeAction): string => {
  if (action === 'BUY') {
    return 'bg-green-100 text-green-800';
  }
  if (action === 'SELL') {
    return 'bg-red-100 text-red-800';
  }
  return 'bg-stone-100 text-stone-700';
};

/** Sortable, filterable trade table with CSV export. */
export default function TradeHistory({
  trades,
  onExport,
  isLoading = false,
}: TradeHistoryProps): JSX.Element {
  const [filter, setFilter] = useState<TradeFilter>('all');
  const [sortField, setSortField] = useState<SortField>('timestamp');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const actionableTrades = useMemo(
    () => trades.filter((trade) => trade.action !== 'HOLD'),
    [trades],
  );
  const filteredTrades = useMemo(() => {
    const nextTrades = actionableTrades.filter(
      (trade) => filter === 'all' || trade.action === filter,
    );
    return [...nextTrades].sort((left, right) => {
      const leftValue = left[sortField];
      const rightValue = right[sortField];
      return compareValues(leftValue, rightValue, sortDirection);
    });
  }, [actionableTrades, filter, sortDirection, sortField]);

  const buyCount = actionableTrades.filter((trade) => trade.action === 'BUY').length;
  const sellCount = actionableTrades.filter((trade) => trade.action === 'SELL').length;

  const handleSort = (field: SortField): void => {
    if (sortField === field) {
      setSortDirection((currentDirection) => (currentDirection === 'asc' ? 'desc' : 'asc'));
      return;
    }
    setSortField(field);
    setSortDirection('desc');
  };

  return (
    <section className="surface-panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 px-5 py-4">
        <div>
          <h3 className="font-semibold text-ink">Trade history</h3>
          <p className="mt-1 text-sm text-stone-600">Showing executed buy and sell records.</p>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={onExport}
          disabled={actionableTrades.length === 0}
        >
          <Download size={16} aria-hidden="true" />
          Export CSV
        </button>
      </div>
      <div className="flex flex-wrap gap-2 border-b border-stone-200 px-5 py-3 text-sm">
        <button
          type="button"
          className={`rounded-md px-3 py-1.5 ${filter === 'all' ? 'bg-blue-100 text-action' : 'bg-stone-100 text-stone-700'}`}
          onClick={() => setFilter('all')}
        >
          All ({actionableTrades.length})
        </button>
        <button
          type="button"
          className={`rounded-md px-3 py-1.5 ${filter === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-stone-100 text-stone-700'}`}
          onClick={() => setFilter('BUY')}
        >
          Buys ({buyCount})
        </button>
        <button
          type="button"
          className={`rounded-md px-3 py-1.5 ${filter === 'SELL' ? 'bg-red-100 text-red-800' : 'bg-stone-100 text-stone-700'}`}
          onClick={() => setFilter('SELL')}
        >
          Sells ({sellCount})
        </button>
      </div>

      {isLoading ? (
        <div className="px-5 py-10 text-center text-sm text-stone-500">Loading trades...</div>
      ) : filteredTrades.length === 0 ? (
        <div className="px-5 py-10 text-center text-sm text-stone-500">
          No executed trades found.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-stone-200 text-sm">
            <thead className="bg-stone-50 text-left text-xs uppercase text-stone-500">
              <tr>
                {sortableHeaders.map((header) => (
                  <th key={header.field} className="px-4 py-3 font-semibold">
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 rounded-md hover:text-ink"
                      onClick={() => handleSort(header.field)}
                    >
                      {header.label}
                      <ArrowDownUp size={12} aria-hidden="true" />
                    </button>
                  </th>
                ))}
                <th className="px-4 py-3 font-semibold">Cost</th>
                <th className="px-4 py-3 font-semibold">Position</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-200 bg-white">
              {filteredTrades.slice(0, 150).map((trade) => (
                <tr key={trade.trade_id} className="hover:bg-stone-50">
                  <td className="whitespace-nowrap px-4 py-3 text-stone-700">
                    {format(new Date(trade.timestamp), 'MMM d, HH:mm')}
                  </td>
                  <td className="px-4 py-3 font-medium text-ink">{trade.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`status-pill ${actionClass(trade.action)}`}>
                      {trade.action}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-stone-700">{trade.quantity.toFixed(4)}</td>
                  <td className="px-4 py-3 text-stone-700">{formatCurrency(trade.price)}</td>
                  <td className="px-4 py-3 text-stone-700">
                    {formatCurrency(trade.portfolio_value)}
                  </td>
                  <td className="px-4 py-3 text-stone-700">{formatCurrency(trade.cost)}</td>
                  <td className="px-4 py-3 text-stone-700">{trade.position_after.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {filteredTrades.length > 150 && (
            <div className="border-t border-stone-200 px-5 py-3 text-center text-sm text-stone-500">
              Showing 150 of {filteredTrades.length} trades.
            </div>
          )}
        </div>
      )}
    </section>
  );
}
