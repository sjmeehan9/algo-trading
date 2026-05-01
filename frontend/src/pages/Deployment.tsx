import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertCircle, LockKeyhole, RefreshCw, Rocket } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

import { deploymentApi, type DeploymentCandidate } from '../api/deployment';
import DeploymentSelectionPanel from '../components/deployment/DeploymentSelectionPanel';
import ModelCandidateList from '../components/deployment/ModelCandidateList';

const toMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to complete deployment request.';

const firstGenerationId = (candidate: DeploymentCandidate | null): string =>
  candidate?.available_generations[0]?.generation_id ?? '';

/** Route page for deployment selection state. */
export default function Deployment(): JSX.Element {
  const queryClient = useQueryClient();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedGenerationId, setSelectedGenerationId] = useState('');

  const candidatesQuery = useQuery({
    queryKey: ['deployment', 'candidates'],
    queryFn: () => deploymentApi.listCandidates(),
  });
  const selectionQuery = useQuery({
    queryKey: ['deployment', 'selection'],
    queryFn: () => deploymentApi.getSelection(),
  });

  const candidates = useMemo(() => candidatesQuery.data ?? [], [candidatesQuery.data]);
  const selectedCandidate = useMemo(
    () => candidates.find((candidate) => candidate.model_id === selectedModelId) ?? null,
    [candidates, selectedModelId],
  );

  useEffect(() => {
    const saved = selectionQuery.data;
    if (saved && candidates.some((candidate) => candidate.model_id === saved.model_id)) {
      setSelectedModelId(saved.model_id);
      setSelectedGenerationId(saved.generation_id);
      return;
    }
    if (!selectedModelId && candidates[0]) {
      setSelectedModelId(candidates[0].model_id);
      setSelectedGenerationId(firstGenerationId(candidates[0]));
    }
  }, [candidates, selectedModelId, selectionQuery.data]);

  useEffect(() => {
    if (!selectedCandidate) {
      return;
    }
    const hasSelectedGeneration = selectedCandidate.available_generations.some(
      (generation) => generation.generation_id === selectedGenerationId,
    );
    if (!hasSelectedGeneration) {
      setSelectedGenerationId(firstGenerationId(selectedCandidate));
    }
  }, [selectedCandidate, selectedGenerationId]);

  const readinessQuery = useQuery({
    queryKey: ['deployment', 'readiness', selectedModelId, selectedGenerationId],
    queryFn: () =>
      deploymentApi.validate({
        model_id: selectedModelId ?? '',
        generation_id: selectedGenerationId,
      }),
    enabled: Boolean(selectedModelId && selectedGenerationId),
  });

  const saveMutation = useMutation({
    mutationFn: () => {
      if (!selectedModelId || !selectedGenerationId) {
        throw new Error('Select a model and generation before saving.');
      }
      return deploymentApi.saveSelection({
        model_id: selectedModelId,
        generation_id: selectedGenerationId,
      });
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['deployment', 'selection'] }),
        queryClient.invalidateQueries({ queryKey: ['deployment', 'candidates'] }),
      ]);
    },
  });

  const clearMutation = useMutation({
    mutationFn: () => deploymentApi.clearSelection(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['deployment', 'selection'] });
    },
  });

  const handleSelectCandidate = (candidate: DeploymentCandidate): void => {
    setSelectedModelId(candidate.model_id);
    setSelectedGenerationId(firstGenerationId(candidate));
  };

  const handleRefresh = (): void => {
    void Promise.all([
      queryClient.invalidateQueries({ queryKey: ['deployment', 'candidates'] }),
      queryClient.invalidateQueries({ queryKey: ['deployment', 'selection'] }),
    ]);
  };

  const errorMessage =
    (saveMutation.error ? toMessage(saveMutation.error) : null) ||
    (clearMutation.error ? toMessage(clearMutation.error) : null) ||
    (candidatesQuery.error ? toMessage(candidatesQuery.error) : null) ||
    (readinessQuery.error ? toMessage(readinessQuery.error) : null);

  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">Deployment</h2>
          <p className="mt-1 text-sm text-stone-600">
            Select a validated core RL generation for Phase 6 trading infrastructure.
          </p>
        </div>
        <button type="button" className="secondary-button" onClick={handleRefresh}>
          <RefreshCw size={16} aria-hidden="true" />
          Refresh
        </button>
      </div>

      {errorMessage && (
        <section
          className="surface-panel flex items-start gap-3 p-4 text-sm text-red-700"
          role="alert"
        >
          <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <div>
            <h3 className="font-semibold">Deployment request failed</h3>
            <p className="mt-1">{errorMessage}</p>
          </div>
        </section>
      )}

      {candidatesQuery.isLoading ? null : candidates.length === 0 ? (
        <section className="surface-panel p-6">
          <div className="flex items-center gap-3">
            <LockKeyhole className="text-action" size={22} aria-hidden="true" />
            <h3 className="font-semibold text-ink">No core RL deployment candidates</h3>
          </div>
          <p className="mt-3 text-sm text-stone-600">
            Create a core RL model, train at least one generation, and run a backtest before saving a
            deployment candidate.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            <Link className="primary-button" to="/models/new">
              New core model
            </Link>
            <Link className="secondary-button" to="/training">
              Training dashboard
            </Link>
            <Link className="secondary-button" to="/backtesting">
              Backtesting
            </Link>
          </div>
        </section>
      ) : null}

      {(candidatesQuery.isLoading || candidates.length > 0) && (
        <section className="grid gap-5 lg:grid-cols-[360px_minmax(0,1fr)]">
          <ModelCandidateList
            candidates={candidates}
            selectedModelId={selectedModelId}
            isLoading={candidatesQuery.isLoading}
            onSelect={handleSelectCandidate}
          />
          <DeploymentSelectionPanel
            candidate={selectedCandidate}
            selectedGenerationId={selectedGenerationId}
            readiness={readinessQuery.data ?? selectedCandidate?.readiness ?? null}
            savedSelection={selectionQuery.data ?? null}
            isValidating={readinessQuery.isFetching}
            isSaving={saveMutation.isPending}
            isClearing={clearMutation.isPending}
            onGenerationChange={setSelectedGenerationId}
            onSave={() => saveMutation.mutate()}
            onClear={() => clearMutation.mutate()}
          />
        </section>
      )}

      <section className="surface-panel p-5">
        <div className="flex items-center gap-3">
          <Rocket className="text-action" size={24} aria-hidden="true" />
          <h3 className="font-semibold text-ink">Saved candidate</h3>
        </div>
        {selectionQuery.data ? (
          <p className="mt-3 text-sm text-stone-600">
            {selectionQuery.data.candidate.name} / Generation {selectionQuery.data.generation_id} /{' '}
            saved {new Date(selectionQuery.data.selected_at).toLocaleString()}
          </p>
        ) : (
          <div className="mt-4 flex items-center gap-3 rounded-md border border-stone-200 bg-stone-50 p-4 text-sm text-stone-600">
            <LockKeyhole size={18} aria-hidden="true" />
            <span>No core RL model generation is saved for deployment.</span>
          </div>
        )}
      </section>
    </div>
  );
}