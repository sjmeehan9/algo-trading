import { apiClient, type PaginatedResponse } from './client';

export type ModelType = 'core_rl' | 'supporting_ml' | 'supporting_rl';

export type SignalType = 'sentiment' | 'trend' | 'volatility' | 'indicator' | 'position' | 'custom';

export type HyperparameterValue = number | string | boolean;

export type ModelState = 'draft' | 'configured' | 'ready' | 'training' | 'failed' | string;

/** Shared model configuration fields returned by the backend. */
export interface ModelConfigBase {
  name: string;
  description?: string | null;
  model_type: ModelType;
  signal_type?: SignalType | null;
  trainer_type: string;
  algorithm: string;
}

/** Payload used to create a model configuration. */
export interface ModelConfigCreate extends ModelConfigBase {
  hyperparameters: Record<string, HyperparameterValue>;
  training_data_config?: Record<string, unknown>;
  supporting_model_ids?: string[];
  strategy_ids?: string[];
  environment_config?: Record<string, unknown>;
  reward_function?: string | null;
  input_data_types?: string[];
  input_frequency?: string | null;
}

/** Payload used to update model configuration fields. */
export type ModelConfigUpdate = Partial<ModelConfigCreate>;

/** Model configuration returned by read and list endpoints. */
export interface ModelConfigResponse extends ModelConfigCreate {
  model_id: string;
  created_at: string;
  updated_at: string;
  state: ModelState;
}

export interface ListModelsParams {
  modelType?: ModelType;
  page?: number;
  pageSize?: number;
}

/** One named readiness check contributing to a supporting model's readiness. */
export interface ReadinessCheck {
  name: string;
  passed: boolean;
  detail?: string | null;
}

/**
 * Lifecycle state snapshot for a supporting model.
 *
 * Mirrors the backend `SupportingModelLifecycleResponse` returned by
 * `GET /api/v1/models/{model_id}/lifecycle`.
 */
export interface SupportingModelLifecycleResponse {
  model_id: string;
  model_type: ModelType;
  state: ModelState;
  model_path?: string | null;
  trainer_class: string;
  input_data_types: string[];
  signal_type: string;
  algorithm?: string | null;
  last_error?: string | null;
  is_ready: boolean;
  readiness_checks: ReadinessCheck[];
}

/** Optional hyperparameter overrides applied when activating a pretrained backend. */
export interface ActivatePretrainedRequest {
  hyperparameters?: Record<string, HyperparameterValue>;
}

/** Request body to load and validate an external supporting model artifact. */
export interface LoadArtifactRequest {
  model_path: string;
}

export interface StrategyInfo {
  strategy_id: string;
  name: string;
  signal_type: SignalType;
  description: string;
  version: string;
  state: string;
}

export interface StrategyDetail extends StrategyInfo {
  filepath: string;
  class_name: string;
}

const toModelListParams = (params: ListModelsParams = {}): Record<string, unknown> => ({
  ...(params.modelType ? { model_type: params.modelType } : {}),
  ...(params.page ? { page: params.page } : {}),
  ...(params.pageSize ? { page_size: params.pageSize } : {}),
});

/** API helpers for model-management endpoints. */
export const modelsApi = {
  list(params?: ListModelsParams): Promise<PaginatedResponse<ModelConfigResponse>> {
    return apiClient.get<PaginatedResponse<ModelConfigResponse>>(
      '/models',
      toModelListParams(params),
    );
  },

  create(config: ModelConfigCreate): Promise<ModelConfigResponse> {
    return apiClient.post<ModelConfigResponse>('/models', config);
  },

  get(modelId: string): Promise<ModelConfigResponse> {
    return apiClient.get<ModelConfigResponse>(`/models/${encodeURIComponent(modelId)}`);
  },

  update(modelId: string, updates: ModelConfigUpdate): Promise<ModelConfigResponse> {
    return apiClient.put<ModelConfigResponse>(`/models/${encodeURIComponent(modelId)}`, updates);
  },

  remove(modelId: string): Promise<void> {
    return apiClient.delete<void>(`/models/${encodeURIComponent(modelId)}`);
  },

  getLifecycle(modelId: string): Promise<SupportingModelLifecycleResponse> {
    return apiClient.get<SupportingModelLifecycleResponse>(
      `/models/${encodeURIComponent(modelId)}/lifecycle`,
    );
  },

  activatePretrained(
    modelId: string,
    request: ActivatePretrainedRequest = {},
  ): Promise<SupportingModelLifecycleResponse> {
    return apiClient.post<SupportingModelLifecycleResponse>(
      `/models/${encodeURIComponent(modelId)}/activate-pretrained`,
      request,
    );
  },

  loadArtifact(
    modelId: string,
    request: LoadArtifactRequest,
  ): Promise<SupportingModelLifecycleResponse> {
    return apiClient.post<SupportingModelLifecycleResponse>(
      `/models/${encodeURIComponent(modelId)}/load-artifact`,
      request,
    );
  },

  unload(modelId: string): Promise<SupportingModelLifecycleResponse> {
    return apiClient.post<SupportingModelLifecycleResponse>(
      `/models/${encodeURIComponent(modelId)}/unload`,
    );
  },
};

/** API helpers for custom strategy catalog endpoints. */
export const strategiesApi = {
  list(signalType?: SignalType): Promise<StrategyInfo[]> {
    return apiClient.get<StrategyInfo[]>(
      '/strategies',
      signalType ? { signal_type: signalType } : undefined,
    );
  },

  get(strategyId: string): Promise<StrategyDetail> {
    return apiClient.get<StrategyDetail>(`/strategies/${encodeURIComponent(strategyId)}`);
  },
};
