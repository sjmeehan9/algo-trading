import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Pause,
  Play,
  Radio,
  RefreshCw,
  ShieldCheck,
  Square,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { deploymentApi } from '../api/deployment';
import { tradingApi, type TradingSessionStatus } from '../api/trading';
import LoadingSpinner from '../components/common/LoadingSpinner';

const BROKERS = [
  { value: 'interactive_brokers', label: 'Interactive Brokers' },
  { value: 'alpaca', label: 'Alpaca' },
] as const;

const statusTone = (status: string): string => {
  switch (status) {
    case 'running':
      return 'bg-green-100 text-green-800';
    case 'paused':
      return 'bg-amber-100 text-amber-800';
    case 'error':
      return 'bg-red-100 text-red-700';
    case 'stopped':
      return 'bg-stone-100 text-stone-700';
    default:
      return 'bg-blue-100 text-blue-800';
  }
};

const toMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to complete trading-session request.';

const parseSymbols = (value: string): string[] =>
  value
    .split(',')
    .map((symbol) => symbol.trim().toUpperCase())
    .filter(Boolean);

const formatNumber = (value: number | null | undefined, digits = 2): string =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';

const formatDateTime = (value: string | null | undefined): string =>
  value ? new Date(value).toLocaleString() : '-';

const latestDecisionText = (session: TradingSessionStatus): string => {
  const decision = session.last_decision;
  if (!decision) {
    return 'No inference decision recorded yet.';
  }

  const action = typeof decision.action === 'string' ? decision.action : 'unknown';
  const confidence = typeof decision.confidence === 'number' ? decision.confidence : null;
  const latency = typeof decision.latency_ms === 'number' ? decision.latency_ms : null;
  return `${action.toUpperCase()} / confidence ${formatNumber(confidence, 3)} / latency ${formatNumber(latency, 1)} ms`;
};

/** Route page for live/paper trading session lifecycle management. */
export default function Trading(): JSX.Element {
  const queryClient = useQueryClient();
  const [modelId, setModelId] = useState('');
  const [generationId, setGenerationId] = useState('');
  const [broker, setBroker] = useState('interactive_brokers');
  const [symbolsInput, setSymbolsInput] = useState('AAPL');
  const [minConfidenceInput, setMinConfidenceInput] = useState('0.6');
  const [maxPositionInput, setMaxPositionInput] = useState('0.1');
  const [closePositionsBySessionId, setClosePositionsBySessionId] = useState<Record<string, boolean>>(
    {},
  );

  const selectionQuery = useQuery({
    queryKey: ['deployment', 'selection', 'trading-page'],
    queryFn: () => deploymentApi.getSelection(),
  });
  const sessionsQuery = useQuery({
    queryKey: ['trading', 'sessions'],
    queryFn: () => tradingApi.listSessions(),
    refetchInterval: 5_000,
  });

  useEffect(() => {
    const selection = selectionQuery.data;
    if (!selection) {
      return;
    }
    setModelId((current) => current || selection.model_id);
    setGenerationId((current) => current || selection.generation_id);
  }, [selectionQuery.data]);

  const sessions = useMemo(() => sessionsQuery.data ?? [], [sessionsQuery.data]);
  const runningSessions = sessions.filter((session) => session.status === 'running');
  const totalDecisions = sessions.reduce((total, session) => total + session.decisions_count, 0);
  const totalExecutedOrders = sessions.reduce(
    (total, session) => total + session.executed_orders_count,
    0,
  );

  const invalidateSessions = async (): Promise<void> => {
    await queryClient.invalidateQueries({ queryKey: ['trading', 'sessions'] });
  };

  const createMutation = useMutation({
    mutationFn: () => {
      const symbols = parseSymbols(symbolsInput);
      const minConfidence = Number(minConfidenceInput);
      const maxPositionPct = Number(maxPositionInput);
      if (!modelId.trim() || !generationId.trim()) {
        throw new Error('Select or enter a model and generation before creating a session.');
      }
      if (symbols.length === 0) {
        throw new Error('Enter at least one trading symbol.');
      }
      if (!Number.isFinite(minConfidence) || minConfidence < 0 || minConfidence > 1) {
        throw new Error('Minimum confidence must be between 0 and 1.');
      }
      if (!Number.isFinite(maxPositionPct) || maxPositionPct <= 0 || maxPositionPct > 1) {
        throw new Error('Maximum position percentage must be greater than 0 and at most 1.');
      }

      return tradingApi.createSession({
        model_id: modelId.trim(),
        generation_id: generationId.trim(),
        mode: 'paper',
        broker,
        symbols,
        supporting_model_ids: selectionQuery.data?.candidate.supporting_model_ids ?? [],
        risk_config: {
          min_confidence: minConfidence,
          max_position_pct: maxPositionPct,
        },
      });
    },
    onSuccess: invalidateSessions,
  });

  const startMutation = useMutation({
    mutationFn: (sessionId: string) => tradingApi.startSession(sessionId),
    onSuccess: invalidateSessions,
  });
  const pauseMutation = useMutation({
    mutationFn: (sessionId: string) => tradingApi.pauseSession(sessionId),
    onSuccess: invalidateSessions,
  });
  const stopMutation = useMutation({
    mutationFn: (sessionId: string) =>
      tradingApi.stopSession(sessionId, Boolean(closePositionsBySessionId[sessionId])),
    onSuccess: invalidateSessions,
  });

  const handleRefresh = (): void => {
    void Promise.all([sessionsQuery.refetch(), selectionQuery.refetch()]);
  };

  const errorMessage =
    (createMutation.error ? toMessage(createMutation.error) : null) ||
    (startMutation.error ? toMessage(startMutation.error) : null) ||
    (pauseMutation.error ? toMessage(pauseMutation.error) : null) ||
    (stopMutation.error ? toMessage(stopMutation.error) : null) ||
    (sessionsQuery.error ? toMessage(sessionsQuery.error) : null) ||
    (selectionQuery.error ? toMessage(selectionQuery.error) : null);

  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">Trading Sessions</h2>
          <p className="mt-1 text-sm text-stone-600">
            Start paper sessions for the saved RL deployment candidate and monitor live decisions.
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
            <h3 className="font-semibold">Trading request failed</h3>
            <p className="mt-1">{errorMessage}</p>
          </div>
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <article className="surface-panel p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-stone-500">Running sessions</p>
              <p className="mt-2 text-3xl font-semibold text-ink">{runningSessions.length}</p>
            </div>
            <Radio className="text-success" size={26} aria-hidden="true" />
          </div>
        </article>
        <article className="surface-panel p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-stone-500">Decisions</p>
              <p className="mt-2 text-3xl font-semibold text-ink">{totalDecisions}</p>
            </div>
            <Activity className="text-action" size={26} aria-hidden="true" />
          </div>
        </article>
        <article className="surface-panel p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-stone-500">Executed orders</p>
              <p className="mt-2 text-3xl font-semibold text-ink">{totalExecutedOrders}</p>
            </div>
            <ShieldCheck className="text-caution" size={26} aria-hidden="true" />
          </div>
        </article>
      </section>

      <section className="surface-panel p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-semibold text-ink">Create paper session</h3>
            <p className="mt-1 text-sm text-stone-600">
              The saved deployment candidate is used when available; manual IDs are accepted for smoke checks.
            </p>
          </div>
          {selectionQuery.data ? (
            <span className="status-pill bg-green-100 text-green-800">
              <CheckCircle2 size={14} aria-hidden="true" />
              Candidate selected
            </span>
          ) : (
            <span className="status-pill bg-amber-100 text-amber-800">No saved candidate</span>
          )}
        </div>

        <form
          className="grid gap-4 xl:grid-cols-[minmax(180px,1fr)_minmax(180px,1fr)_180px_180px_150px_150px_auto]"
          onSubmit={(event) => {
            event.preventDefault();
            createMutation.mutate();
          }}
        >
          <div>
            <label htmlFor="session-model-id" className="mb-1 block text-sm font-medium text-ink">
              Model ID
            </label>
            <input
              id="session-model-id"
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
              className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label
              htmlFor="session-generation-id"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Generation ID
            </label>
            <input
              id="session-generation-id"
              value={generationId}
              onChange={(event) => setGenerationId(event.target.value)}
              className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label htmlFor="session-broker" className="mb-1 block text-sm font-medium text-ink">
              Broker
            </label>
            <select
              id="session-broker"
              value={broker}
              onChange={(event) => setBroker(event.target.value)}
              className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
            >
              {BROKERS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="session-symbols" className="mb-1 block text-sm font-medium text-ink">
              Symbols
            </label>
            <input
              id="session-symbols"
              value={symbolsInput}
              onChange={(event) => setSymbolsInput(event.target.value)}
              className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label
              htmlFor="session-min-confidence"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Min confidence
            </label>
            <input
              id="session-min-confidence"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minConfidenceInput}
              onChange={(event) => setMinConfidenceInput(event.target.value)}
              className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label
              htmlFor="session-max-position"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Max position
            </label>
            <input
              id="session-max-position"
              type="number"
              min={0.01}
              max={1}
              step={0.01}
              value={maxPositionInput}
              onChange={(event) => setMaxPositionInput(event.target.value)}
              className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
            />
          </div>
          <div className="flex items-end">
            <button type="submit" className="primary-button w-full" disabled={createMutation.isPending}>
              <Play size={16} aria-hidden="true" />
              {createMutation.isPending ? 'Creating' : 'Create'}
            </button>
          </div>
        </form>
      </section>

      <section className="surface-panel overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 px-5 py-4">
          <div>
            <h3 className="font-semibold text-ink">Live session dashboard</h3>
            <p className="mt-1 text-sm text-stone-600">
              Review lifecycle state, decisions, positions, and order activity.
            </p>
          </div>
          <span className="status-pill bg-stone-100 text-stone-600">{sessions.length} sessions</span>
        </div>

        {sessionsQuery.isLoading ? (
          <LoadingSpinner />
        ) : sessions.length === 0 ? (
          <div className="px-5 py-12 text-center text-sm text-stone-500">
            No trading sessions have been created.
          </div>
        ) : (
          <div className="divide-y divide-stone-200">
            {sessions.map((session) => {
              const canStart = session.status === 'created' || session.status === 'paused';
              const canPause = session.status === 'running';
              const canStop = session.status !== 'stopped' && session.status !== 'stopping';
              const closePositions = Boolean(closePositionsBySessionId[session.session_id]);

              return (
                <article key={session.session_id} className="space-y-4 p-5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h4 className="break-all font-semibold text-ink">{session.session_id}</h4>
                        <span className={`status-pill ${statusTone(session.status)}`}>
                          {session.status}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-stone-600">
                        {session.broker} / {session.mode} / {session.symbols.join(', ')}
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={!canStart || startMutation.isPending}
                        onClick={() => startMutation.mutate(session.session_id)}
                      >
                        <Play size={16} aria-hidden="true" />
                        Start
                      </button>
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={!canPause || pauseMutation.isPending}
                        onClick={() => pauseMutation.mutate(session.session_id)}
                      >
                        <Pause size={16} aria-hidden="true" />
                        Pause
                      </button>
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={!canStop || stopMutation.isPending}
                        onClick={() => stopMutation.mutate(session.session_id)}
                      >
                        <Square size={16} aria-hidden="true" />
                        Stop
                      </button>
                    </div>
                  </div>

                  <dl className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
                    <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
                      <dt className="text-stone-500">Started</dt>
                      <dd className="mt-1 font-semibold text-ink">{formatDateTime(session.started_at)}</dd>
                    </div>
                    <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
                      <dt className="text-stone-500">Last price</dt>
                      <dd className="mt-1 font-semibold text-ink">{formatNumber(session.last_price)}</dd>
                    </div>
                    <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
                      <dt className="text-stone-500">Decisions</dt>
                      <dd className="mt-1 font-semibold text-ink">{session.decisions_count}</dd>
                    </div>
                    <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
                      <dt className="text-stone-500">Orders</dt>
                      <dd className="mt-1 font-semibold text-ink">
                        {session.executed_orders_count}/{session.orders_count}
                      </dd>
                    </div>
                  </dl>

                  <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
                    <div className="rounded-md border border-stone-200 p-3 text-sm">
                      <h5 className="font-semibold text-ink">Latest decision</h5>
                      <p className="mt-2 text-stone-600">{latestDecisionText(session)}</p>
                      {session.error_message && (
                        <p className="mt-2 text-red-700">{session.error_message}</p>
                      )}
                    </div>
                    <div className="rounded-md border border-stone-200 p-3 text-sm">
                      <div className="flex items-center justify-between gap-3">
                        <h5 className="font-semibold text-ink">Positions</h5>
                        <label className="flex items-center gap-2 text-xs text-stone-600">
                          <input
                            type="checkbox"
                            checked={closePositions}
                            onChange={(event) =>
                              setClosePositionsBySessionId((current) => ({
                                ...current,
                                [session.session_id]: event.target.checked,
                              }))
                            }
                            className="h-4 w-4 rounded border-stone-300 text-action"
                          />
                          Close on stop
                        </label>
                      </div>
                      {Object.keys(session.positions).length === 0 ? (
                        <p className="mt-2 text-stone-600">No tracked positions.</p>
                      ) : (
                        <div className="mt-2 space-y-1">
                          {Object.entries(session.positions).map(([symbol, quantity]) => (
                            <div key={symbol} className="flex justify-between gap-3">
                              <span>{symbol}</span>
                              <span className="font-semibold">{quantity}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
