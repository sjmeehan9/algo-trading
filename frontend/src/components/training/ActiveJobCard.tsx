import { Activity, CircleStop, Clock, Gauge, TrendingUp } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import type { TrainingJob, TrainingMetrics } from '../../api/training';
import { useTrainingProgress } from '../../hooks/useTrainingProgress';
import TrainingMetricsChart from './TrainingMetricsChart';

interface ActiveJobCardProps {
  job: TrainingJob;
  modelName: string;
  isCancelling?: boolean;
  onCancel: (jobId: string) => void;
}

const clampPercent = (value: number): number => Math.min(100, Math.max(0, value));

const getMetric = (metrics: TrainingMetrics, keys: string[]): number | undefined => {
  for (const key of keys) {
    const value = metrics[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
};

const formatDuration = (seconds: number): string => {
  const roundedSeconds = Math.max(0, Math.round(seconds));
  const hours = Math.floor(roundedSeconds / 3600);
  const minutes = Math.floor((roundedSeconds % 3600) / 60);
  const remainingSeconds = roundedSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
};

const formatMetric = (value: number | undefined, digits = 2): string =>
  value === undefined ? '-' : value.toFixed(digits);

/** Card displaying the currently running training job and its live metrics. */
export default function ActiveJobCard({
  job,
  modelName,
  isCancelling = false,
  onCancel,
}: ActiveJobCardProps): JSX.Element {
  const { progress, metricsHistory } = useTrainingProgress(job.model_id, job.job_id);
  const activeProgress = progress ?? job;
  const progressPercent = clampPercent(activeProgress.progress_percent);
  const metrics = activeProgress.current_metrics ?? {};
  const reward = getMetric(metrics, ['episode_reward', 'reward', 'mean_reward', 'final_reward']);
  const loss = getMetric(metrics, ['loss', 'policy_loss', 'value_loss']);
  const episodeLength = getMetric(metrics, ['episode_length', 'episodes_completed']);
  const learningRate = getMetric(metrics, ['learning_rate']);
  const startedAt = job.started_at ? new Date(job.started_at) : null;

  return (
    <section className="surface-panel overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-stone-200 px-5 py-4">
        <div className="flex items-start gap-3">
          <span className="mt-1 rounded-md bg-blue-50 p-2 text-action">
            <Activity size={20} aria-hidden="true" />
          </span>
          <div>
            <h3 className="text-lg font-semibold text-ink">{modelName}</h3>
            <p className="mt-1 text-sm text-stone-600">
              {startedAt ? `Started ${formatDistanceToNow(startedAt)} ago` : 'Preparing to start'}
            </p>
            {job.description && <p className="mt-1 text-sm text-stone-500">{job.description}</p>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="status-pill bg-blue-100 text-blue-800">{activeProgress.status}</span>
          <button
            type="button"
            className="secondary-button border-red-200 text-red-700 hover:bg-red-50"
            disabled={isCancelling}
            onClick={() => onCancel(job.job_id)}
          >
            <CircleStop size={16} aria-hidden="true" />
            {isCancelling ? 'Cancelling' : 'Cancel'}
          </button>
        </div>
      </div>

      <div className="space-y-5 px-5 py-5">
        <div>
          <div className="mb-2 flex flex-wrap justify-between gap-2 text-sm text-stone-700">
            <span className="font-medium">Progress: {progressPercent.toFixed(1)}%</span>
            <span>
              {activeProgress.current_timestep.toLocaleString()} /{' '}
              {activeProgress.total_timesteps.toLocaleString()} timesteps
            </span>
          </div>
          <div
            className="h-3 overflow-hidden rounded-full bg-stone-200"
            role="progressbar"
            aria-label={`Training progress for ${modelName}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progressPercent}
          >
            <div
              className="h-full rounded-full bg-action transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          {progress?.estimated_remaining_seconds ? (
            <p className="mt-2 flex items-center gap-1 text-sm text-stone-500">
              <Clock size={14} aria-hidden="true" />
              {formatDuration(progress.estimated_remaining_seconds)} remaining
            </p>
          ) : null}
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-md border border-stone-200 bg-stone-50 px-4 py-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase text-stone-500">
              <TrendingUp size={14} aria-hidden="true" />
              Reward
            </div>
            <div className="mt-2 text-2xl font-semibold text-ink">{formatMetric(reward)}</div>
          </div>
          <div className="rounded-md border border-stone-200 bg-stone-50 px-4 py-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase text-stone-500">
              <Gauge size={14} aria-hidden="true" />
              Loss
            </div>
            <div className="mt-2 text-2xl font-semibold text-ink">{formatMetric(loss, 4)}</div>
          </div>
          <div className="rounded-md border border-stone-200 bg-stone-50 px-4 py-3">
            <div className="text-xs font-semibold uppercase text-stone-500">Episode length</div>
            <div className="mt-2 text-2xl font-semibold text-ink">
              {episodeLength === undefined ? '-' : Math.round(episodeLength).toLocaleString()}
            </div>
          </div>
          <div className="rounded-md border border-stone-200 bg-stone-50 px-4 py-3">
            <div className="text-xs font-semibold uppercase text-stone-500">Learning rate</div>
            <div className="mt-2 text-2xl font-semibold text-ink">
              {learningRate === undefined ? '-' : learningRate.toExponential(2)}
            </div>
          </div>
        </div>

        <div className="h-72 rounded-md border border-stone-200 p-3">
          <TrainingMetricsChart data={metricsHistory} />
        </div>
      </div>
    </section>
  );
}
