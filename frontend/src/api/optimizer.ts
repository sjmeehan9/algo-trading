import { apiClient } from './client';
import type { HyperparameterValue, ModelConfigResponse } from './models';

export type SuggestionConfidence = 'low' | 'medium' | 'high';
export type OptimizationSource = 'openai' | 'fallback';
export type SuggestionOutcomeStatus = 'not_applied' | 'pending_next_generation' | 'evaluated';

export interface OptimizerAnalyzeRequest {
  model_id: string;
  max_generations?: number;
  include_backtest_metrics?: boolean;
}

export interface HyperparameterSuggestion {
  suggestion_id: string;
  parameter: string;
  current_value?: HyperparameterValue | null;
  suggested_value: HyperparameterValue;
  rationale: string;
  confidence: SuggestionConfidence;
  expected_impact: string;
  applied: boolean;
  applied_at?: string | null;
  outcome_status: SuggestionOutcomeStatus;
  outcome_notes?: string | null;
}

export interface OptimizationResult {
  result_id: string;
  model_id: string;
  created_at: string;
  analyzed_generation_ids: string[];
  suggestions: HyperparameterSuggestion[];
  priority_changes: string[];
  summary: string;
  source: OptimizationSource;
  raw_response?: string | null;
}

export interface ApplySuggestionRequest {
  model_id: string;
  suggestion_id: string;
  result_id?: string | null;
}

export interface AppliedSuggestionResponse {
  suggestion: HyperparameterSuggestion;
  updated_model: ModelConfigResponse;
  optimization_result: OptimizationResult;
}

/** API helpers for LLM-assisted hyperparameter optimization. */
export const optimizerApi = {
  analyze(request: OptimizerAnalyzeRequest): Promise<OptimizationResult> {
    return apiClient.post<OptimizationResult>('/optimizer/analyze', request);
  },

  getLatest(modelId: string): Promise<OptimizationResult | null> {
    return apiClient.get<OptimizationResult | null>(
      `/optimizer/suggestions/${encodeURIComponent(modelId)}`,
    );
  },

  apply(request: ApplySuggestionRequest): Promise<AppliedSuggestionResponse> {
    return apiClient.post<AppliedSuggestionResponse>('/optimizer/apply', request);
  },
};
