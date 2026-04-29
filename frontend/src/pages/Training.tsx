import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertCircle, CircleSlash2, Play, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

import { modelsApi, type ModelConfigResponse } from '../api/models';
import { trainingApi, type GenerationSummary, type TrainingJob } from '../api/training';
import ActiveJobCard from '../components/training/ActiveJobCard';
import GenerationComparisonChart from '../components/training/GenerationComparisonChart';
import GenerationHistory from '../components/training/GenerationHistory';
import JobQueue from '../components/training/JobQueue';
import LoadingSpinner from '../components/common/LoadingSpinner';

const DEFAULT_TIMESTEPS = 100_000;
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

const toMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to complete training request.';

const createModelNameMap = (models: ModelConfigResponse[]): Map<string, string> =>
  new Map(models.map((model) => [model.model_id, model.name]));

const getModelName = (modelNames: Map<string, string>, modelId: string): string =>
  modelNames.get(modelId) ?? modelId;

const getSelectedGenerations = (
  generations: GenerationSummary[],
  selectedGenerationIds: string[],
): GenerationSummary[] =>
  selectedGenerationIds
    .map((generationId) =>
      generations.find((generation) => generation.generation_id === generationId),
    )
    .filter((generation): generation is GenerationSummary => Boolean(generation));

/** Route page for training job monitoring and generation review. */
export default function Training(): JSX.Element {
  const queryClient = useQueryClient();
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [timestepsInput, setTimestepsInput] = useState(String(DEFAULT_TIMESTEPS));
  const [description, setDescription] = useState('');
  const [selectedGenerationIds, setSelectedGenerationIds] = useState<string[]>([]);

  const modelsQuery = useQuery({
    queryKey: ['models', 'core_rl', 'training-dashboard'],
    queryFn: () => modelsApi.list({ modelType: 'core_rl', pageSize: 100 }),
  });
  const runningJobsQuery = useQuery({
    queryKey: ['training-jobs', 'running'],
    queryFn: () => trainingApi.listJobs({ status: 'running' }),
    refetchInterval: 5_000,
  });
  const queuedJobsQuery = useQuery({
    queryKey: ['training-jobs', 'queued'],
    queryFn: () => trainingApi.listJobs({ status: 'queued' }),
    refetchInterval: 5_000,
  });
  const recentJobsQuery = useQuery({
    queryKey: ['training-jobs', 'recent'],
    queryFn: () => trainingApi.listJobs(),
    refetchInterval: 10_000,
  });
  const generationsQuery = useQuery({
    queryKey: ['generations', selectedModelId],
    queryFn: () => trainingApi.listGenerations(selectedModelId).then((response) => response.items),
    enabled: Boolean(selectedModelId),
  });

  const models = useMemo(() => modelsQuery.data?.items ?? [], [modelsQuery.data]);
  const modelNames = useMemo(() => createModelNameMap(models), [models]);
  const activeJobs = runningJobsQuery.data ?? [];
  const queuedJobs = queuedJobsQuery.data ?? [];
  const generations = useMemo(() => generationsQuery.data ?? [], [generationsQuery.data]);
  const selectedGenerations = useMemo(
    () => getSelectedGenerations(generations, selectedGenerationIds),
    [generations, selectedGenerationIds],
  );
  const recentTerminalJobs = useMemo(
    () =>
      (recentJobsQuery.data ?? []).filter((job) => TERMINAL_STATUSES.has(job.status)).slice(0, 5),
    [recentJobsQuery.data],
  );

  useEffect(() => {
    if (!selectedModelId && models[0]) {
      setSelectedModelId(models[0].model_id);
    }
  }, [models, selectedModelId]);

  const invalidateTrainingData = async (modelId?: string): Promise<void> => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['training-jobs'] }),
      queryClient.invalidateQueries({ queryKey: ['training-jobs', 'running'] }),
      queryClient.invalidateQueries({ queryKey: ['training-jobs', 'queued'] }),
      queryClient.invalidateQueries({ queryKey: ['training-jobs', 'recent'] }),
      ...(modelId ? [queryClient.invalidateQueries({ queryKey: ['generations', modelId] })] : []),
    ]);
  };

  const startMutation = useMutation({
    mutationFn: () => {
      const totalTimesteps = Number(timestepsInput);
      if (!selectedModelId) {
        throw new Error('Select a model before starting training.');
      }
      if (!Number.isInteger(totalTimesteps) || totalTimesteps < 1000) {
        throw new Error('Total timesteps must be at least 1,000.');
      }
      return trainingApi.createJob({
        model_id: selectedModelId,
        total_timesteps: totalTimesteps,
        description: description.trim() || undefined,
      });
    },
    onSuccess: async (job) => {
      setDescription('');
      await invalidateTrainingData(job.model_id);
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (job: TrainingJob) => trainingApi.cancelJob(job.job_id),
    onSuccess: async (job) => {
      await invalidateTrainingData(job.model_id);
    },
  });

  const reorderMutation = useMutation({
    mutationFn: (jobIds: string[]) => trainingApi.reorderQueue(jobIds),
    onSuccess: async () => {
      await invalidateTrainingData(selectedModelId || undefined);
    },
  });

  const handleModelChange = (modelId: string): void => {
    setSelectedModelId(modelId);
    setSelectedGenerationIds([]);
  };

  const handleRefresh = (): void => {
    void invalidateTrainingData(selectedModelId || undefined);
  };

  const hasModelLoadError = Boolean(modelsQuery.error);
  const hasJobLoadError = Boolean(runningJobsQuery.error || queuedJobsQuery.error);
  const startError = startMutation.error ? toMessage(startMutation.error) : null;
  const cancelError = cancelMutation.error ? toMessage(cancelMutation.error) : null;
  const reorderError = reorderMutation.error ? toMessage(reorderMutation.error) : null;

  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">Training Dashboard</h2>
          <p className="mt-1 text-sm text-stone-600">
            Monitor live jobs, manage the queue, and compare model generations.
          </p>
        </div>
        <button type="button" className="secondary-button" onClick={handleRefresh}>
          <RefreshCw size={16} aria-hidden="true" />
          Refresh
        </button>
      </div>

      {(hasModelLoadError || hasJobLoadError || startError || cancelError || reorderError) && (
        <section
          className="surface-panel flex items-start gap-3 p-4 text-sm text-red-700"
          role="alert"
        >
          <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <div>
            <h3 className="font-semibold">Training dashboard request failed</h3>
            <p className="mt-1">
              {startError ||
                cancelError ||
                reorderError ||
                toMessage(modelsQuery.error || runningJobsQuery.error || queuedJobsQuery.error)}
            </p>
          </div>
        </section>
      )}

      <section className="surface-panel p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-semibold text-ink">Start training</h3>
            <p className="mt-1 text-sm text-stone-600">
              Queue a new generation for a configured core RL model.
            </p>
          </div>
          <Link className="secondary-button" to="/models/new">
            New core model
          </Link>
        </div>

        {modelsQuery.isLoading ? (
          <LoadingSpinner />
        ) : models.length === 0 ? (
          <div className="rounded-md border border-dashed border-stone-300 px-4 py-8 text-center text-sm text-stone-500">
            No core RL model configurations are available.
          </div>
        ) : (
          <form
            className="grid gap-4 lg:grid-cols-[minmax(220px,1.2fr)_180px_minmax(220px,1fr)_auto]"
            onSubmit={(event) => {
              event.preventDefault();
              startMutation.mutate();
            }}
          >
            <div>
              <label htmlFor="training-model" className="mb-1 block text-sm font-medium text-ink">
                Training model
              </label>
              <select
                id="training-model"
                value={selectedModelId}
                onChange={(event) => handleModelChange(event.target.value)}
                className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
              >
                {models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label
                htmlFor="training-timesteps"
                className="mb-1 block text-sm font-medium text-ink"
              >
                Timesteps
              </label>
              <input
                id="training-timesteps"
                type="number"
                min={1000}
                step={1000}
                value={timestepsInput}
                onChange={(event) => setTimestepsInput(event.target.value)}
                className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label
                htmlFor="training-description"
                className="mb-1 block text-sm font-medium text-ink"
              >
                Description
              </label>
              <input
                id="training-description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
                placeholder="Experiment label"
              />
            </div>
            <div className="flex items-end">
              <button
                type="submit"
                className="primary-button w-full"
                disabled={startMutation.isPending || !selectedModelId}
              >
                <Play size={16} aria-hidden="true" />
                {startMutation.isPending ? 'Queueing' : 'Start training'}
              </button>
            </div>
          </form>
        )}
      </section>

      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-ink">Active training</h3>
          <span className="status-pill bg-stone-100 text-stone-600">
            {activeJobs.length} running
          </span>
        </div>
        {runningJobsQuery.isLoading ? (
          <LoadingSpinner />
        ) : activeJobs.length > 0 ? (
          <div className="space-y-4">
            {activeJobs.map((job) => (
              <ActiveJobCard
                key={job.job_id}
                job={job}
                modelName={getModelName(modelNames, job.model_id)}
                isCancelling={cancelMutation.variables?.job_id === job.job_id}
                onCancel={() => cancelMutation.mutate(job)}
              />
            ))}
          </div>
        ) : (
          <section className="surface-panel flex min-h-44 flex-col items-center justify-center text-center text-stone-500">
            <CircleSlash2 size={28} aria-hidden="true" />
            <p className="mt-3 text-sm">No active training job.</p>
          </section>
        )}
      </section>

      <JobQueue
        jobs={queuedJobs}
        modelNames={modelNames}
        cancellingJobId={cancelMutation.variables?.job_id}
        isReordering={reorderMutation.isPending}
        onCancel={(jobId) => {
          const job = queuedJobs.find((queuedJob) => queuedJob.job_id === jobId);
          if (job) {
            cancelMutation.mutate(job);
          }
        }}
        onReorder={(jobIds) => reorderMutation.mutate(jobIds)}
      />

      <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h3 className="text-lg font-semibold text-ink">Generation history</h3>
              <p className="mt-1 text-sm text-stone-600">
                Select two to five generations to compare reward and evaluation metrics.
              </p>
            </div>
            <select
              value={selectedModelId}
              onChange={(event) => handleModelChange(event.target.value)}
              className="min-w-56 rounded-md border border-stone-300 px-3 py-2 text-sm"
              aria-label="Generation model"
            >
              {models.length === 0 && <option value="">No models</option>}
              {models.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          {generationsQuery.isLoading ? (
            <LoadingSpinner />
          ) : (
            <GenerationHistory
              generations={generations}
              selectedIds={selectedGenerationIds}
              onSelectionChange={setSelectedGenerationIds}
            />
          )}
          <GenerationComparisonChart generations={selectedGenerations} />
        </div>

        <section className="surface-panel overflow-hidden">
          <div className="border-b border-stone-200 px-5 py-4">
            <h3 className="font-semibold text-ink">Recent outcomes</h3>
          </div>
          {recentJobsQuery.isLoading ? (
            <div className="p-5">
              <LoadingSpinner />
            </div>
          ) : recentTerminalJobs.length === 0 ? (
            <div className="px-5 py-10 text-center text-sm text-stone-500">
              No completed jobs yet.
            </div>
          ) : (
            <div className="divide-y divide-stone-200">
              {recentTerminalJobs.map((job) => (
                <div key={job.job_id} className="px-5 py-4 text-sm">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium text-ink">
                        {getModelName(modelNames, job.model_id)}
                      </div>
                      <div className="mt-1 text-xs text-stone-500">{job.job_id}</div>
                    </div>
                    <span
                      className={`status-pill ${
                        job.status === 'completed'
                          ? 'bg-green-100 text-green-800'
                          : job.status === 'failed'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-stone-100 text-stone-700'
                      }`}
                    >
                      {job.status}
                    </span>
                  </div>
                  {job.error_message && (
                    <p className="mt-2 text-xs text-red-700">{job.error_message}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </section>
    </div>
  );
}
