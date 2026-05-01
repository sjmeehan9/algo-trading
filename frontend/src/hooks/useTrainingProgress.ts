import { useCallback, useEffect, useState } from 'react';

import { wsClient } from '../api/websocket';
import type { TrainingJobStatus, TrainingMetrics } from '../api/training';

const MAX_HISTORY_POINTS = 500;
const TERMINAL_STATUSES = new Set<TrainingJobStatus>(['completed', 'failed', 'cancelled']);

export interface TrainingProgress {
  job_id: string;
  model_id: string;
  status: TrainingJobStatus;
  progress_percent: number;
  current_timestep: number;
  total_timesteps: number;
  current_metrics: TrainingMetrics;
  estimated_remaining_seconds?: number | null;
  timestamp?: string;
}

export interface TrainingMetricsHistoryPoint {
  timestep: number;
  reward?: number;
  loss?: number;
  learningRate?: number;
}

export interface UseTrainingProgressResult {
  progress: TrainingProgress | null;
  metricsHistory: TrainingMetricsHistoryPoint[];
  isComplete: boolean;
  clearHistory: () => void;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const getNumberMetric = (metrics: TrainingMetrics, keys: string[]): number | undefined => {
  for (const key of keys) {
    const value = metrics[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
};

const normalizeStatus = (value: unknown): TrainingJobStatus | null => {
  if (
    value === 'queued' ||
    value === 'running' ||
    value === 'completed' ||
    value === 'failed' ||
    value === 'cancelled'
  ) {
    return value;
  }
  return null;
};

const normalizeProgress = (value: unknown): TrainingProgress | null => {
  if (!isRecord(value)) {
    return null;
  }

  const status = normalizeStatus(value.status);
  if (
    typeof value.job_id !== 'string' ||
    typeof value.model_id !== 'string' ||
    status === null ||
    typeof value.progress_percent !== 'number' ||
    typeof value.current_timestep !== 'number' ||
    typeof value.total_timesteps !== 'number'
  ) {
    return null;
  }

  return {
    job_id: value.job_id,
    model_id: value.model_id,
    status,
    progress_percent: value.progress_percent,
    current_timestep: value.current_timestep,
    total_timesteps: value.total_timesteps,
    current_metrics: isRecord(value.current_metrics) ? value.current_metrics : {},
    estimated_remaining_seconds:
      typeof value.estimated_remaining_seconds === 'number'
        ? value.estimated_remaining_seconds
        : null,
    timestamp: typeof value.timestamp === 'string' ? value.timestamp : undefined,
  };
};

/** Convert a progress payload into a chartable metrics-history point. */
export const trainingProgressToHistoryPoint = (
  progress: TrainingProgress,
): TrainingMetricsHistoryPoint | null => {
  const reward = getNumberMetric(progress.current_metrics, [
    'episode_reward',
    'reward',
    'mean_reward',
    'final_reward',
  ]);
  const loss = getNumberMetric(progress.current_metrics, ['loss', 'policy_loss', 'value_loss']);
  const learningRate = getNumberMetric(progress.current_metrics, ['learning_rate']);

  if (reward === undefined && loss === undefined && learningRate === undefined) {
    return null;
  }

  return {
    timestep: progress.current_timestep,
    ...(reward !== undefined ? { reward } : {}),
    ...(loss !== undefined ? { loss } : {}),
    ...(learningRate !== undefined ? { learningRate } : {}),
  };
};

/** Subscribe to training progress for a model or specific job via WebSocket. */
export function useTrainingProgress(
  modelId: string | null,
  jobId?: string | null,
): UseTrainingProgressResult {
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetricsHistoryPoint[]>([]);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (!modelId) {
      return undefined;
    }

    const topic = `training:${jobId || modelId}`;
    const handleMessage = (data: unknown): void => {
      const incomingProgress = normalizeProgress(data);
      if (!incomingProgress) {
        return;
      }
      if (incomingProgress.model_id !== modelId) {
        return;
      }
      if (jobId && incomingProgress.job_id !== jobId) {
        return;
      }

      setProgress(incomingProgress);
      setIsComplete(TERMINAL_STATUSES.has(incomingProgress.status));

      const historyPoint = trainingProgressToHistoryPoint(incomingProgress);
      if (historyPoint) {
        setMetricsHistory((currentHistory) => [
          ...currentHistory.slice(-(MAX_HISTORY_POINTS - 1)),
          historyPoint,
        ]);
      }
    };

    const unsubscribe = wsClient.subscribe<TrainingProgress>(topic, handleMessage);

    return () => {
      unsubscribe();
      setProgress(null);
      setMetricsHistory([]);
      setIsComplete(false);
    };
  }, [jobId, modelId]);

  const clearHistory = useCallback(() => {
    setMetricsHistory([]);
  }, []);

  return { progress, metricsHistory, isComplete, clearHistory };
}
