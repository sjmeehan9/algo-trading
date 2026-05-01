import { apiClient } from './client';
import type { PerformanceMetrics } from './backtesting';

export type ReadinessCheckStatus = 'passed' | 'warning' | 'failed';

export interface ReadinessCheck {
  key: string;
  label: string;
  status: ReadinessCheckStatus;
  message: string;
}

export interface DeploymentReadiness {
  deployable: boolean;
  errors: string[];
  warnings: string[];
  checks: ReadinessCheck[];
}

export interface DeploymentValidationRequest {
  model_id: string;
  generation_id: string;
}

export interface DeploymentBacktestSummary {
  source: string;
  backtest_id?: string | null;
  completed_at?: string | null;
  metrics: PerformanceMetrics | Record<string, number | null>;
}

export interface DeploymentGenerationSummary {
  generation_id: string;
  generation_number: number;
  status: string;
  created_at: string;
  training_duration_seconds: number;
  final_reward?: number | null;
  model_path?: string | null;
  metrics: Record<string, number | string | boolean | null | undefined>;
  evaluation?: DeploymentBacktestSummary | null;
}

export interface DeploymentCandidate {
  model_id: string;
  name: string;
  description?: string | null;
  algorithm: string;
  trainer_type: string;
  state: string;
  supporting_model_ids: string[];
  strategy_ids: string[];
  available_generations: DeploymentGenerationSummary[];
  latest_backtest?: DeploymentBacktestSummary | null;
  readiness: DeploymentReadiness;
}

export interface DeploymentSelection {
  model_id: string;
  generation_id: string;
  selected_at: string;
  candidate: DeploymentCandidate;
  readiness: DeploymentReadiness;
  selected_by?: string | null;
}

/** API helpers for pre-deployment candidate selection. */
export const deploymentApi = {
  listCandidates(): Promise<DeploymentCandidate[]> {
    return apiClient.get<DeploymentCandidate[]>('/deployment/candidates');
  },

  validate(request: DeploymentValidationRequest): Promise<DeploymentReadiness> {
    return apiClient.post<DeploymentReadiness>('/deployment/validate', request);
  },

  getSelection(): Promise<DeploymentSelection | null> {
    return apiClient.get<DeploymentSelection | null>('/deployment/selection');
  },

  saveSelection(request: DeploymentValidationRequest): Promise<DeploymentSelection> {
    return apiClient.put<DeploymentSelection>('/deployment/selection', request);
  },

  clearSelection(): Promise<{ status: string }> {
    return apiClient.delete<{ status: string }>('/deployment/selection');
  },
};