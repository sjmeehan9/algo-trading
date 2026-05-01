import { apiClient, type PaginatedResponse } from './client';

export type BacktestStatus = 'running' | 'completed' | 'failed';
export type TradeAction = 'BUY' | 'SELL' | 'HOLD';

export interface BacktestRequest {
  model_id: string;
  generation_id: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  symbols?: string[] | null;
  include_transaction_costs: boolean;
  description?: string | null;
}

export interface TradeRecord {
  trade_id: number;
  timestamp: string;
  symbol: string;
  action: TradeAction;
  quantity: number;
  price: number;
  cost: number;
  position_after: number;
  portfolio_value: number;
  signal_confidence?: number | null;
}

export interface PerformanceMetrics {
  total_return: number;
  total_return_dollars: number;
  annualized_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  max_drawdown_duration_days: number;
  volatility: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  average_win: number;
  average_loss: number;
  largest_win: number;
  largest_loss: number;
  average_trade_duration: number;
  exposure_time: number;
}

export interface EquityPoint {
  timestamp: string;
  portfolio_value: number;
  cash: number;
  position_value: number;
  drawdown: number;
}

export interface BacktestResult {
  backtest_id: string;
  model_id: string;
  generation_id: string;
  status: BacktestStatus;
  created_at: string;
  completed_at?: string | null;
  request: BacktestRequest;
  metrics?: PerformanceMetrics | null;
  equity_curve: EquityPoint[];
  error_message?: string | null;
}

export interface BacktestComparison {
  backtests: BacktestResult[];
  metric_comparison: Record<string, Array<number | null>>;
  best_by_metric: Record<string, string>;
}

export interface ListBacktestsParams {
  modelId?: string;
  generationId?: string;
  status?: BacktestStatus;
  page?: number;
  pageSize?: number;
}

const toListBacktestParams = (params: ListBacktestsParams = {}): Record<string, unknown> => ({
  ...(params.modelId ? { model_id: params.modelId } : {}),
  ...(params.generationId ? { generation_id: params.generationId } : {}),
  ...(params.status ? { status: params.status } : {}),
  ...(params.page ? { page: params.page } : {}),
  ...(params.pageSize ? { page_size: params.pageSize } : {}),
});

/** API helpers for model backtesting and evaluation endpoints. */
export const backtestingApi = {
  run(request: BacktestRequest): Promise<BacktestResult> {
    return apiClient.post<BacktestResult>('/backtests', request);
  },

  list(params?: ListBacktestsParams): Promise<PaginatedResponse<BacktestResult>> {
    return apiClient.get<PaginatedResponse<BacktestResult>>(
      '/backtests',
      toListBacktestParams(params),
    );
  },

  get(backtestId: string): Promise<BacktestResult> {
    return apiClient.get<BacktestResult>(`/backtests/${encodeURIComponent(backtestId)}`);
  },

  getTrades(backtestId: string): Promise<TradeRecord[]> {
    return apiClient.get<TradeRecord[]>(`/backtests/${encodeURIComponent(backtestId)}/trades`);
  },

  compare(backtestIds: string[]): Promise<BacktestComparison> {
    return apiClient.post<BacktestComparison>('/backtests/compare', { backtest_ids: backtestIds });
  },
};
