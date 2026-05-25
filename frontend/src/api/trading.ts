import { apiClient } from './client';

export type TradingSessionStatusValue =
  | 'created'
  | 'starting'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'stopped'
  | 'error';

export interface TradingSessionCreateRequest {
  model_id: string;
  generation_id: string;
  mode: 'paper' | 'live';
  broker: string;
  symbols: string[];
  supporting_model_ids?: string[];
  risk_config?: Record<string, number | string | boolean>;
}

export interface TradingSessionStatus {
  session_id: string;
  status: TradingSessionStatusValue | string;
  model_id: string;
  generation_id: string;
  broker: string;
  mode: string;
  symbols: string[];
  supporting_model_ids: string[];
  created_at: string;
  started_at?: string | null;
  stopped_at?: string | null;
  positions: Record<string, number>;
  orders_count: number;
  decisions_count: number;
  executed_orders_count: number;
  error_count: number;
  error_message?: string | null;
  last_price?: number | null;
  last_decision?: Record<string, unknown> | null;
}

/** API helpers for Phase 6 trading-session lifecycle endpoints. */
export const tradingApi = {
  listSessions(status?: string): Promise<TradingSessionStatus[]> {
    return apiClient.get<TradingSessionStatus[]>(
      '/trading/sessions',
      status ? { status } : undefined,
    );
  },

  createSession(request: TradingSessionCreateRequest): Promise<TradingSessionStatus> {
    return apiClient.post<TradingSessionStatus>('/trading/sessions', request);
  },

  startSession(sessionId: string): Promise<TradingSessionStatus> {
    return apiClient.post<TradingSessionStatus>(
      `/trading/sessions/${encodeURIComponent(sessionId)}/start`,
    );
  },

  pauseSession(sessionId: string): Promise<TradingSessionStatus> {
    return apiClient.post<TradingSessionStatus>(
      `/trading/sessions/${encodeURIComponent(sessionId)}/pause`,
    );
  },

  stopSession(sessionId: string, closePositions = false): Promise<TradingSessionStatus> {
    return apiClient.post<TradingSessionStatus>(
      `/trading/sessions/${encodeURIComponent(sessionId)}/stop`,
      { close_positions: closePositions },
    );
  },
};
