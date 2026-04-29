import { ArrowDown, ArrowUp, CircleStop, GripVertical } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import type { TrainingJob } from '../../api/training';

interface JobQueueProps {
  jobs: TrainingJob[];
  modelNames: Map<string, string>;
  cancellingJobId?: string;
  isReordering?: boolean;
  onCancel: (jobId: string) => void;
  onReorder: (jobIds: string[]) => void;
}

const moveJob = (jobs: TrainingJob[], fromIndex: number, toIndex: number): string[] => {
  const reordered = [...jobs];
  const [movedJob] = reordered.splice(fromIndex, 1);
  if (!movedJob) {
    return jobs.map((job) => job.job_id);
  }
  reordered.splice(toIndex, 0, movedJob);
  return reordered.map((job) => job.job_id);
};

/** Queue table with cancellation and order controls for queued jobs. */
export default function JobQueue({
  jobs,
  modelNames,
  cancellingJobId,
  isReordering = false,
  onCancel,
  onReorder,
}: JobQueueProps): JSX.Element {
  if (jobs.length === 0) {
    return (
      <section className="surface-panel px-5 py-10 text-center text-sm text-stone-500">
        No queued training jobs.
      </section>
    );
  }

  return (
    <section className="surface-panel overflow-hidden">
      <div className="border-b border-stone-200 px-5 py-4">
        <h3 className="font-semibold text-ink">Queued jobs</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-stone-200 text-sm">
          <thead className="bg-stone-50 text-left text-xs uppercase text-stone-500">
            <tr>
              <th className="px-4 py-3 font-semibold">Order</th>
              <th className="px-4 py-3 font-semibold">Model</th>
              <th className="px-4 py-3 font-semibold">Timesteps</th>
              <th className="px-4 py-3 font-semibold">Queued</th>
              <th className="px-4 py-3 font-semibold">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-200 bg-white">
            {jobs.map((job, index) => {
              const modelName = modelNames.get(job.model_id) ?? job.model_id;
              const isCancelling = cancellingJobId === job.job_id;
              return (
                <tr key={job.job_id}>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 text-stone-500">
                      <GripVertical size={16} aria-hidden="true" />
                      <span className="font-medium text-ink">{index + 1}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="font-medium text-ink">{modelName}</div>
                    {job.description && (
                      <div className="mt-1 max-w-md truncate text-xs text-stone-500">
                        {job.description}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-stone-700">
                    {job.total_timesteps.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-stone-600">
                    {formatDistanceToNow(new Date(job.created_at))} ago
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        className="icon-button h-8 w-8"
                        disabled={index === 0 || isReordering}
                        aria-label={`Move ${modelName} up`}
                        onClick={() => onReorder(moveJob(jobs, index, index - 1))}
                      >
                        <ArrowUp size={15} aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        className="icon-button h-8 w-8"
                        disabled={index === jobs.length - 1 || isReordering}
                        aria-label={`Move ${modelName} down`}
                        onClick={() => onReorder(moveJob(jobs, index, index + 1))}
                      >
                        <ArrowDown size={15} aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        className="icon-button h-8 w-8 border-red-200 text-red-700 hover:bg-red-50"
                        disabled={isCancelling}
                        aria-label={`Cancel queued job for ${modelName}`}
                        onClick={() => onCancel(job.job_id)}
                      >
                        <CircleStop size={15} aria-hidden="true" />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
