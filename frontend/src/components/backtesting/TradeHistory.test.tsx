import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { TradeRecord } from '../../api/backtesting';
import TradeHistory from './TradeHistory';

const trades: TradeRecord[] = [
  {
    trade_id: 1,
    timestamp: '2026-01-01T14:30:00Z',
    symbol: 'AAPL',
    action: 'BUY',
    quantity: 12,
    price: 100,
    cost: 1.2,
    position_after: 12,
    portfolio_value: 100_000,
  },
  {
    trade_id: 2,
    timestamp: '2026-01-02T14:30:00Z',
    symbol: 'AAPL',
    action: 'SELL',
    quantity: 12,
    price: 108,
    cost: 1.3,
    position_after: 0,
    portfolio_value: 101_200,
  },
  {
    trade_id: 3,
    timestamp: '2026-01-03T14:30:00Z',
    symbol: 'AAPL',
    action: 'HOLD',
    quantity: 0,
    price: 110,
    cost: 0,
    position_after: 0,
    portfolio_value: 101_200,
  },
];

describe('TradeHistory', () => {
  it('filters executable trades and calls CSV export', async () => {
    const user = userEvent.setup();
    const onExport = vi.fn();

    render(<TradeHistory trades={trades} onExport={onExport} />);

    expect(screen.getByRole('button', { name: /All \(2\)/ })).toBeInTheDocument();
    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('SELL')).toBeInTheDocument();
    expect(screen.queryByText('HOLD')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /Buys \(1\)/ }));

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.queryByText('SELL')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /Export CSV/ }));
    expect(onExport).toHaveBeenCalledTimes(1);
  });
});
