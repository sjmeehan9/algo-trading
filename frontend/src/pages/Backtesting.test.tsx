import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  backtestingApi,
  type BacktestComparison,
  type BacktestResult,
  type TradeRecord,
} from '../api/backtesting';
import type { PaginatedResponse } from '../api/client';
import { modelsApi, type ModelConfigResponse } from '../api/models';
import { trainingApi, type GenerationSummary } from '../api/training';
import Backtesting from './Backtesting';

vi.mock('../api/models', () => ({
  modelsApi: {
    list: vi.fn(),
  },
}));

vi.mock('../api/training', () => ({
  trainingApi: {
    listGenerations: vi.fn(),
  },
}));

vi.mock('../api/backtesting', () => ({
  backtestingApi: {
    run: vi.fn(),
    list: vi.fn(),
    get: vi.fn(),
    getTrades: vi.fn(),
    compare: vi.fn(),
  },
}));

const coreModel: ModelConfigResponse = {
  model_id: 'core-1',
  name: 'Core PPO',
  description: 'Primary policy',
  model_type: 'core_rl',
  signal_type: null,
  trainer_type: 'stable_baselines3',
  algorithm: 'ppo',
  hyperparameters: { learning_rate: 0.0003 },
  training_data_config: { symbols: ['AAPL'] },
  supporting_model_ids: [],
  strategy_ids: [],
  environment_config: {},
  reward_function: 'profit_seeker',
  input_data_types: [],
  input_frequency: null,
  created_at: '2026-04-01T00:00:00Z',
  updated_at: '2026-04-02T00:00:00Z',
  state: 'configured',
};

const generations: GenerationSummary[] = [
  {
    generation_id: 'gen-2',
    generation_number: 2,
    model_id: 'core-1',
    created_at: '2026-04-29T08:00:00Z',
    training_duration_seconds: 1200,
    final_reward: 16.5,
    metrics: { sharpe_ratio: 1.2, total_return: 0.08 },
  },
];

const trades: TradeRecord[] = [
  {
    trade_id: 1,
    timestamp: '2026-01-01T14:30:00Z',
    symbol: 'AAPL',
    action: 'BUY',
    quantity: 100,
    price: 100,
    cost: 0,
    position_after: 100,
    portfolio_value: 100_000,
  },
  {
    trade_id: 2,
    timestamp: '2026-01-03T14:30:00Z',
    symbol: 'AAPL',
    action: 'SELL',
    quantity: 100,
    price: 112,
    cost: 0,
    position_after: 0,
    portfolio_value: 104_000,
  },
];

const firstResult: BacktestResult = {
  backtest_id: 'backtest-1',
  model_id: 'core-1',
  generation_id: 'gen-2',
  status: 'completed',
  created_at: '2026-04-29T10:00:00Z',
  completed_at: '2026-04-29T10:00:05Z',
  request: {
    model_id: 'core-1',
    generation_id: 'gen-2',
    start_date: '2026-01-01',
    end_date: '2026-01-03',
    initial_capital: 100_000,
    symbols: ['AAPL'],
    include_transaction_costs: true,
    description: 'January evaluation',
  },
  metrics: {
    total_return: 0.04,
    total_return_dollars: 4000,
    annualized_return: 0.14,
    sharpe_ratio: 1.42,
    sortino_ratio: 1.9,
    max_drawdown: 0.03,
    max_drawdown_duration_days: 2,
    volatility: 0.18,
    win_rate: 0.5,
    profit_factor: 1.7,
    total_trades: 2,
    winning_trades: 1,
    losing_trades: 1,
    average_win: 1200,
    average_loss: -200,
    largest_win: 1200,
    largest_loss: -200,
    average_trade_duration: 12,
    exposure_time: 0.6,
  },
  equity_curve: [
    {
      timestamp: '2026-01-01T14:30:00Z',
      portfolio_value: 100_000,
      cash: 100_000,
      position_value: 0,
      drawdown: 0,
    },
    {
      timestamp: '2026-01-03T14:30:00Z',
      portfolio_value: 104_000,
      cash: 104_000,
      position_value: 0,
      drawdown: -0.01,
    },
  ],
  error_message: null,
};

const secondResult: BacktestResult = {
  ...firstResult,
  backtest_id: 'backtest-2',
  request: {
    ...firstResult.request,
    description: 'Second evaluation',
  },
  metrics: firstResult.metrics
    ? {
        ...firstResult.metrics,
        total_return: 0.02,
        sharpe_ratio: 0.8,
      }
    : null,
};

const comparison: BacktestComparison = {
  backtests: [firstResult, secondResult],
  metric_comparison: {
    total_return: [0.04, 0.02],
    sharpe_ratio: [1.42, 0.8],
  },
  best_by_metric: {
    total_return: 'backtest-1',
    sharpe_ratio: 'backtest-1',
  },
};

const createPaginated = <Item,>(items: Item[]): PaginatedResponse<Item> => ({
  items,
  total: items.length,
  page: 1,
  page_size: 100,
  pages: items.length > 0 ? 1 : 0,
});

const renderBacktesting = (): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Backtesting />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

describe('Backtesting', () => {
  beforeEach(() => {
    vi.mocked(modelsApi.list).mockResolvedValue(createPaginated([coreModel]));
    vi.mocked(trainingApi.listGenerations).mockResolvedValue(createPaginated(generations));
    vi.mocked(backtestingApi.list).mockResolvedValue(createPaginated([firstResult, secondResult]));
    vi.mocked(backtestingApi.run).mockResolvedValue(firstResult);
    vi.mocked(backtestingApi.getTrades).mockResolvedValue(trades);
    vi.mocked(backtestingApi.compare).mockResolvedValue(comparison);
  });

  it('runs a backtest and displays metrics, charts, and trades', async () => {
    const user = userEvent.setup();
    renderBacktesting();

    await screen.findByLabelText('Model');
    await screen.findByRole('option', { name: /Generation 2/ });
    await user.selectOptions(screen.getByLabelText('Generation'), 'gen-2');
    await user.type(screen.getByLabelText('Description'), 'January evaluation');
    const runButton = screen.getByRole('button', { name: /Run backtest/ });
    await waitFor(() => expect(runButton).not.toBeDisabled());
    await user.click(runButton);

    await waitFor(() => expect(backtestingApi.run).toHaveBeenCalledTimes(1));
    expect(backtestingApi.run).toHaveBeenCalledWith(
      expect.objectContaining({
        model_id: 'core-1',
        generation_id: 'gen-2',
        initial_capital: 100_000,
        symbols: ['AAPL'],
        description: 'January evaluation',
      }),
    );
    expect(backtestingApi.getTrades).toHaveBeenCalledWith('backtest-1');
    expect(await screen.findAllByText('backtest-1')).not.toHaveLength(0);
    expect(screen.getByText('Total return')).toBeInTheDocument();
    expect(screen.getByText('4.00%')).toBeInTheDocument();
    expect(screen.getByText('Equity curve')).toBeInTheDocument();
    expect(screen.getByText('Drawdown')).toBeInTheDocument();
    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('SELL')).toBeInTheDocument();
  });

  it('loads stored trades and compares selected results', async () => {
    const user = userEvent.setup();
    renderBacktesting();

    await user.click(await screen.findByRole('button', { name: 'backtest-2' }));
    await waitFor(() => expect(backtestingApi.getTrades).toHaveBeenCalledWith('backtest-2'));

    await user.click(screen.getByRole('checkbox', { name: /Compare backtest-1/ }));
    await user.click(screen.getByRole('checkbox', { name: /Compare backtest-2/ }));
    await user.click(screen.getByRole('button', { name: /Compare selected \(2\)/ }));

    await waitFor(() =>
      expect(backtestingApi.compare).toHaveBeenCalledWith(['backtest-1', 'backtest-2']),
    );
    expect(await screen.findByRole('heading', { name: /Backtest comparison/ })).toBeInTheDocument();
    expect(screen.getAllByText('Sharpe')).not.toHaveLength(0);
  });
});
