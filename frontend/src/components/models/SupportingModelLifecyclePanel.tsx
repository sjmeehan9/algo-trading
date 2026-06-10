import {
  AlertCircle,
  CheckCircle2,
  CircleSlash2,
  FolderInput,
  PlayCircle,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import { useState } from 'react';

import type { ModelConfigResponse, ModelState, ReadinessCheck } from '../../api/models';
import {
  supportsPretrainedActivation,
  useSupportingModelLifecycle,
} from '../../hooks/useSupportingModelForm';
import LoadingSpinner from '../common/LoadingSpinner';

interface SupportingModelLifecyclePanelProps {
  /** The persisted supporting model whose lifecycle is being managed. */
  model: ModelConfigResponse;
}

const READY_PILL = 'bg-green-100 text-green-800';
const ERROR_PILL = 'bg-red-100 text-red-700';
const NEUTRAL_PILL = 'bg-stone-100 text-stone-700';

/** Return the status-pill class for a lifecycle state. */
const statePillClass = (state: ModelState): string => {
  const normalized = String(state).toLowerCase();
  if (normalized === 'ready') {
    return READY_PILL;
  }
  if (normalized === 'error' || normalized === 'failed') {
    return ERROR_PILL;
  }
  return NEUTRAL_PILL;
};

/** Render one readiness check row with a pass/fail icon and detail. */
const ReadinessCheckRow = ({ check }: { check: ReadinessCheck }): JSX.Element => (
  <li className="flex items-start gap-2 text-sm">
    {check.passed ? (
      <CheckCircle2 className="mt-0.5 shrink-0 text-green-600" size={16} aria-hidden="true" />
    ) : (
      <XCircle className="mt-0.5 shrink-0 text-stone-400" size={16} aria-hidden="true" />
    )}
    <span className="min-w-0">
      <span className="font-medium text-ink">{check.name}</span>
      {check.detail && <span className="ml-1 text-stone-500">— {check.detail}</span>}
    </span>
  </li>
);

/**
 * Operator panel for moving a supporting model to `ready`.
 *
 * Surfaces lifecycle state, current artifact path, last error, input data types,
 * signal type, and readiness checks, and exposes the three readiness actions
 * defined in `docs/errors/test-3-3-gap.md`: activate a pretrained sentiment
 * backend, load an external artifact path, and unload. The shown state always
 * reflects the backend response (never an optimistic `ready`), action controls
 * are disabled while a request is in flight, and backend lifecycle errors are
 * surfaced inline.
 */
export default function SupportingModelLifecyclePanel({
  model,
}: SupportingModelLifecyclePanelProps): JSX.Element {
  const {
    lifecycle,
    isLoading,
    loadError,
    actionError,
    isMutating,
    activatePretrained,
    loadArtifact,
    unload,
    refetch,
  } = useSupportingModelLifecycle(model.model_id);
  const [artifactPath, setArtifactPath] = useState('');
  const [validationMessage, setValidationMessage] = useState<string | null>(null);

  // Prefer the live lifecycle snapshot; fall back to the configured model while
  // the first lifecycle request resolves so the panel is never blank.
  const state: ModelState = lifecycle?.state ?? model.state;
  const algorithm = lifecycle?.algorithm ?? model.algorithm;
  const signalType = lifecycle?.signal_type ?? model.signal_type ?? '—';
  const inputDataTypes = lifecycle?.input_data_types ?? model.input_data_types ?? [];
  const modelPath = lifecycle?.model_path ?? null;
  const lastError = lifecycle?.last_error ?? null;
  const isReady = lifecycle?.is_ready ?? String(state).toLowerCase() === 'ready';
  const canActivatePretrained = supportsPretrainedActivation(algorithm);

  // Mutation rejections are already captured into `actionError` by React Query;
  // swallow the rejected promise here so it does not surface as an unhandled
  // rejection while still letting the inline error render.
  const ignoreRejection = (): void => undefined;

  const handleActivate = (): void => {
    setValidationMessage(null);
    activatePretrained().catch(ignoreRejection);
  };

  const handleLoadArtifact = (): void => {
    const trimmed = artifactPath.trim();
    if (!trimmed) {
      setValidationMessage('Enter an artifact path before loading.');
      return;
    }
    setValidationMessage(null);
    loadArtifact(trimmed).catch(ignoreRejection);
  };

  const handleUnload = (): void => {
    setValidationMessage(null);
    unload().catch(ignoreRejection);
  };

  return (
    <section className="surface-panel p-5" aria-label="Supporting model lifecycle">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-ink">Lifecycle &amp; Readiness</h3>
          <p className="mt-1 text-sm text-stone-600">
            Make this supporting model <span className="font-medium">ready</span> by activating a
            pretrained backend, loading an external artifact, or training it from the Training
            dashboard.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`status-pill ${statePillClass(state)}`}>{String(state)}</span>
          <button
            type="button"
            className="secondary-button"
            onClick={() => refetch()}
            disabled={isMutating}
          >
            <RefreshCw size={16} aria-hidden="true" />
            Refresh
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="mt-4">
          <LoadingSpinner />
        </div>
      ) : loadError ? (
        <div
          className="mt-4 flex items-start gap-3 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          role="alert"
        >
          <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <div>
            <p className="font-semibold">Unable to load lifecycle state</p>
            <p className="mt-1">{loadError}</p>
          </div>
        </div>
      ) : (
        <>
          <dl className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Signal type
              </dt>
              <dd className="mt-1 text-sm text-ink">{signalType}</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Algorithm
              </dt>
              <dd className="mt-1 text-sm text-ink">{algorithm ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Input data types
              </dt>
              <dd className="mt-1 flex flex-wrap gap-1.5">
                {inputDataTypes.length > 0 ? (
                  inputDataTypes.map((dataType) => (
                    <span
                      key={dataType}
                      className="rounded-md bg-stone-100 px-1.5 py-0.5 text-xs font-medium text-stone-600"
                    >
                      {dataType}
                    </span>
                  ))
                ) : (
                  <span className="text-sm text-stone-500">—</span>
                )}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Current artifact path
              </dt>
              <dd className="mt-1 break-all text-sm text-ink">
                {modelPath ?? <span className="text-stone-500">No artifact loaded</span>}
              </dd>
            </div>
          </dl>

          {lastError && (
            <div
              className="mt-4 flex items-start gap-3 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              role="alert"
            >
              <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
              <div>
                <p className="font-semibold">Last lifecycle error</p>
                <p className="mt-1 break-words">{lastError}</p>
              </div>
            </div>
          )}

          {lifecycle && lifecycle.readiness_checks.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-semibold text-ink">Readiness checks</h4>
              <ul className="mt-2 space-y-1.5">
                {lifecycle.readiness_checks.map((check) => (
                  <ReadinessCheckRow key={check.name} check={check} />
                ))}
              </ul>
            </div>
          )}

          {(actionError || validationMessage) && (
            <div
              className="mt-4 flex items-start gap-3 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              role="alert"
            >
              <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
              <p className="break-words">{actionError ?? validationMessage}</p>
            </div>
          )}

          <div className="mt-5 space-y-4 border-t border-stone-200 pt-5">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                className="primary-button"
                onClick={handleActivate}
                disabled={isMutating || !canActivatePretrained}
                title={
                  canActivatePretrained
                    ? undefined
                    : 'Pretrained activation is only available for sentiment ML models.'
                }
              >
                <PlayCircle size={16} aria-hidden="true" />
                Activate pretrained
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={handleUnload}
                disabled={isMutating || !isReady}
                title={isReady ? undefined : 'Only a ready model can be unloaded.'}
              >
                <CircleSlash2 size={16} aria-hidden="true" />
                Unload
              </button>
            </div>
            {!canActivatePretrained && (
              <p className="text-xs text-stone-500">
                Pretrained activation supports sentiment backends only. Train this model from the
                Training dashboard or load an external artifact below.
              </p>
            )}

            <div>
              <label
                htmlFor="lifecycle-artifact-path"
                className="mb-1 block text-sm font-medium text-ink"
              >
                Load external artifact
              </label>
              <div className="flex flex-wrap gap-2">
                <input
                  id="lifecycle-artifact-path"
                  value={artifactPath}
                  onChange={(event) => setArtifactPath(event.target.value)}
                  placeholder="data/models/news_sentiment or /path/to/model.zip"
                  className="min-w-[16rem] flex-1 rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                  disabled={isMutating}
                />
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleLoadArtifact}
                  disabled={isMutating}
                >
                  <FolderInput size={16} aria-hidden="true" />
                  Load artifact
                </button>
              </div>
              <p className="mt-1 text-xs text-stone-500">
                The API validates and loads the artifact; the model only becomes ready after the
                backend load succeeds.
              </p>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
