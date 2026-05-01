import { describe, expect, it } from 'vitest';

import type { TradeRecord } from '../api/backtesting';
import { tradesToCsv } from './useBacktest';

describe('tradesToCsv', () => {
  it('serializes trade records into escaped CSV content', () => {
    const trades: TradeRecord[] = [
      {
        trade_id: 1,
        timestamp: '2026-01-01T14:30:00Z',
        symbol: 'AAPL',
        action: 'BUY',
        quantity: 10,
        price: 101.25,
        cost: 1.5,
        position_after: 10,
        portfolio_value: 100_000,
        signal_confidence: 0.88,
      },
      {
        trade_id: 2,
        timestamp: '2026-01-02T14:30:00Z',
        symbol: 'MSFT,TEST',
        action: 'SELL',
        quantity: 5,
        price: 205.5,
        cost: 1,
        position_after: 0,
        portfolio_value: 101_000,
        signal_confidence: null,
      },
    ];

    const csv = tradesToCsv(trades);

    expect(csv.split('\n')[0]).toBe(
      'trade_id,timestamp,symbol,action,quantity,price,cost,position_after,portfolio_value,signal_confidence',
    );
    expect(csv).toContain('1,2026-01-01T14:30:00Z,AAPL,BUY,10,101.25,1.5,10,100000,0.88');
    expect(csv).toContain('2,2026-01-02T14:30:00Z,"MSFT,TEST",SELL,5,205.5,1,0,101000,');
  });
});
