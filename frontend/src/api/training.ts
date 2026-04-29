import { apiClient, type PaginatedResponse } from './client';

export type TrainingJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TrainingMetrics {
  episode_reward?: number;
  episode_length?: number;
  reward?: number;
  mean_reward?: number;
  final_reward?: number;
  loss?: number;
  policy_loss?: number;
  value_loss?: number;
  learning_rate?: number;
  [key: string]: unknown;
}

export interface TrainingJob {
  job_id: string;
  model_id: string;
  status: TrainingJobStatus;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  progress_percent: number;
  current_timestep: number;
  total_timesteps: number;
  current_metrics: TrainingMetrics;
  error_message?: string | null;
  generation_id?: string | null;
  description?: string | null;
  training_config: Record<string, unknown>;
  data_config: Record<string, unknown>;
}

export interface TrainingJobCreate {
  model_id: string;
  training_config?: Record<string, unknown>;
  data_config?: Record<string, unknown>;
  total_timesteps?: number;
  description?: string;
}

export interface ListTrainingJobsParams {
  status?: TrainingJobStatus;
  modelId?: string;
}

export interface GenerationSummary {
  generation_id: string;
  generation_number: number;
  model_id: string;
  created_at: string;
  training_duration_seconds: number;
  final_reward?: number | null;
  metrics: Record<string, number | string | boolean | null | undefined>;
}

const toListJobParams = (params: ListTrainingJobsParams = {}): Record<string, unknown> => ({
  ...(params.status ? { status: params.status } : {}),
  ...(params.modelId ? { model_id: params.modelId } : {}),
});

/** API helpers for training control and generation history endpoints. */
export const trainingApi = {
  listJobs(params?: ListTrainingJobsParams): Promise<TrainingJob[]> {
    return apiClient.get<TrainingJob[]>('/training/jobs', toListJobParams(params));
  },

  createJob(request: TrainingJobCreate): Promise<TrainingJob> {
    return apiClient.post<TrainingJob>('/training/jobs', request);
  },

  cancelJob(jobId: string): Promise<TrainingJob> {
    return apiClient.post<TrainingJob>(`/training/jobs/${encodeURIComponent(jobId)}/cancel`);
  },

  reorderQueue(jobIds: string[]): Promise<TrainingJob[]> {
    return apiClient.post<TrainingJob[]>('/training/jobs/reorder', { job_ids: jobIds });
  },

  listGenerations(
    modelId: string,
    page = 1,
    pageSize = 100,
  ): Promise<PaginatedResponse<GenerationSummary>> {
    return apiClient.get<PaginatedResponse<GenerationSummary>>(
      `/models/${encodeURIComponent(modelId)}/generations`,
      { page, page_size: pageSize },
    );
  },
};
