import { BrainCircuit, CheckCircle2 } from 'lucide-react';

import type { DeploymentCandidate } from '../../api/deployment';
import LoadingSpinner from '../common/LoadingSpinner';

interface ModelCandidateListProps {
  candidates: DeploymentCandidate[];
  selectedModelId: string | null;
  isLoading?: boolean;
  onSelect: (candidate: DeploymentCandidate) => void;
}

/** Selectable list of core RL deployment candidates. */
export default function ModelCandidateList({
  candidates,
  selectedModelId,
  isLoading = false,
  onSelect,
}: ModelCandidateListProps): JSX.Element {
  if (isLoading) {
    return (
      <section className="surface-panel p-5">
        <LoadingSpinner />
      </section>
    );
  }

  if (candidates.length === 0) {
    return (
      <section className="surface-panel p-5 text-center text-sm text-stone-500">
        <BrainCircuit className="mx-auto mb-3 text-stone-400" size={30} aria-hidden="true" />
        <p>No deployable root candidates found.</p>
      </section>
    );
  }

  return (
    <section className="surface-panel overflow-hidden">
      <div className="border-b border-stone-200 px-5 py-4">
        <h3 className="font-semibold text-ink">Core RL candidates</h3>
      </div>
      <div className="divide-y divide-stone-200">
        {candidates.map((candidate) => {
          const isSelected = selectedModelId === candidate.model_id;
          const generationCount = candidate.available_generations.length;

          return (
            <button
              key={candidate.model_id}
              type="button"
              className={`flex w-full items-start justify-between gap-4 px-5 py-4 text-left transition hover:bg-stone-50 ${
                isSelected ? 'bg-blue-50' : 'bg-white'
              }`}
              onClick={() => onSelect(candidate)}
            >
              <span className="min-w-0">
                <span className="block font-semibold text-ink">{candidate.name}</span>
                <span className="mt-1 block text-xs uppercase tracking-normal text-stone-500">
                  {candidate.algorithm} / {candidate.trainer_type}
                </span>
                <span className="mt-2 block text-sm text-stone-600">
                  {generationCount} generation{generationCount === 1 ? '' : 's'} /{' '}
                  {candidate.supporting_model_ids.length} supporting input
                  {candidate.supporting_model_ids.length === 1 ? '' : 's'}
                </span>
              </span>
              <span
                className={`status-pill shrink-0 ${
                  candidate.readiness.deployable
                    ? 'bg-green-100 text-green-800'
                    : 'bg-stone-100 text-stone-700'
                }`}
              >
                {candidate.readiness.deployable && (
                  <CheckCircle2 className="mr-1" size={13} aria-hidden="true" />
                )}
                {candidate.readiness.deployable ? 'Ready' : 'Review'}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}