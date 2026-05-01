import { Save, Trash2 } from 'lucide-react';

import type {
  DeploymentCandidate,
  DeploymentReadiness,
  DeploymentSelection,
} from '../../api/deployment';
import GenerationSelector from './GenerationSelector';
import ReadinessChecklist from './ReadinessChecklist';
import SupportingInputsSummary from './SupportingInputsSummary';

interface DeploymentSelectionPanelProps {
  candidate: DeploymentCandidate | null;
  selectedGenerationId: string;
  readiness: DeploymentReadiness | null;
  savedSelection: DeploymentSelection | null;
  isValidating?: boolean;
  isSaving?: boolean;
  isClearing?: boolean;
  onGenerationChange: (generationId: string) => void;
  onSave: () => void;
  onClear: () => void;
}

const formatPercentMetric = (value: unknown): string =>
  typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '-';

const getMetric = (candidate: DeploymentCandidate | null, metricName: string): unknown => {
  const metrics = candidate?.latest_backtest?.metrics;
  if (!metrics || typeof metrics !== 'object') {
    return null;
  }
  return (metrics as Record<string, unknown>)[metricName];
};

/** Detail panel for validating and persisting one deployment selection. */
export default function DeploymentSelectionPanel({
  candidate,
  selectedGenerationId,
  readiness,
  savedSelection,
  isValidating = false,
  isSaving = false,
  isClearing = false,
  onGenerationChange,
  onSave,
  onClear,
}: DeploymentSelectionPanelProps): JSX.Element {
  if (!candidate) {
    return (
      <section className="surface-panel flex min-h-96 items-center justify-center p-8 text-center text-sm text-stone-500">
        Select a core RL model candidate to review deployment readiness.
      </section>
    );
  }

  const savedMatches =
    savedSelection?.model_id === candidate.model_id && savedSelection.generation_id === selectedGenerationId;
  const canSave = Boolean(readiness?.deployable && selectedGenerationId && !isSaving);

  return (
    <div className="space-y-5">
      <section className="surface-panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-ink">{candidate.name}</h3>
            <p className="mt-1 text-sm text-stone-600">
              {candidate.algorithm} / {candidate.trainer_type} / {candidate.state}
            </p>
            {candidate.description && <p className="mt-2 text-sm text-stone-700">{candidate.description}</p>}
          </div>
          {savedMatches && <span className="status-pill bg-blue-100 text-action">Saved</span>}
        </div>

        <div className="mt-5">
          <GenerationSelector
            generations={candidate.available_generations}
            selectedGenerationId={selectedGenerationId}
            onChange={onGenerationChange}
          />
        </div>

        <dl className="mt-5 grid gap-3 text-sm sm:grid-cols-3">
          <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
            <dt className="text-stone-500">Total return</dt>
            <dd className="mt-1 font-semibold text-ink">
              {formatPercentMetric(getMetric(candidate, 'total_return'))}
            </dd>
          </div>
          <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
            <dt className="text-stone-500">Max drawdown</dt>
            <dd className="mt-1 font-semibold text-ink">
              {formatPercentMetric(getMetric(candidate, 'max_drawdown'))}
            </dd>
          </div>
          <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
            <dt className="text-stone-500">Sharpe</dt>
            <dd className="mt-1 font-semibold text-ink">
              {typeof getMetric(candidate, 'sharpe_ratio') === 'number'
                ? (getMetric(candidate, 'sharpe_ratio') as number).toFixed(2)
                : '-'}
            </dd>
          </div>
        </dl>

        <div className="mt-5 flex flex-wrap gap-2">
          <button type="button" className="primary-button" disabled={!canSave} onClick={onSave}>
            <Save size={16} aria-hidden="true" />
            {isSaving ? 'Saving' : 'Save candidate'}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={!savedSelection || isClearing}
            onClick={onClear}
          >
            <Trash2 size={16} aria-hidden="true" />
            {isClearing ? 'Clearing' : 'Clear saved'}
          </button>
        </div>
      </section>

      <ReadinessChecklist readiness={readiness} isLoading={isValidating} />
      <SupportingInputsSummary
        supportingModelIds={candidate.supporting_model_ids}
        strategyIds={candidate.strategy_ids}
      />
    </div>
  );
}