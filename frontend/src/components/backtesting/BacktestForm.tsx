import { useQuery } from '@tanstack/react-query';
import { CalendarRange, Play, Plus, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Controller, useForm } from 'react-hook-form';

import { modelsApi, type ModelConfigResponse } from '../../api/models';
import { trainingApi, type GenerationSummary } from '../../api/training';

export interface BacktestFormData {
  model_id: string;
  generation_id: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  symbols: string[];
  include_transaction_costs: boolean;
  description: string;
}

interface BacktestFormProps {
  onSubmit: (data: BacktestFormData) => void;
  isLoading: boolean;
}

const DEFAULT_INITIAL_CAPITAL = 100_000;
const DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'SPY'];

const todayIso = (): string => new Date().toISOString().slice(0, 10);

const daysAgoIso = (days: number): string => {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().slice(0, 10);
};

const normalizeSymbol = (value: string): string => value.trim().toUpperCase();

const modelOptionLabel = (model: ModelConfigResponse): string =>
  `${model.name} (${model.algorithm.toUpperCase()})`;

const generationOptionLabel = (generation: GenerationSummary): string => {
  const reward =
    typeof generation.final_reward === 'number'
      ? `, reward ${generation.final_reward.toFixed(2)}`
      : '';
  return `Generation ${generation.generation_number}${reward}`;
};

/** Configuration form for running generation backtests. */
export default function BacktestForm({ onSubmit, isLoading }: BacktestFormProps): JSX.Element {
  const [symbolInput, setSymbolInput] = useState('');
  const {
    control,
    handleSubmit,
    resetField,
    setValue,
    watch,
    formState: { errors },
  } = useForm<BacktestFormData>({
    defaultValues: {
      model_id: '',
      generation_id: '',
      start_date: daysAgoIso(90),
      end_date: todayIso(),
      initial_capital: DEFAULT_INITIAL_CAPITAL,
      symbols: ['AAPL'],
      include_transaction_costs: true,
      description: '',
    },
  });

  const selectedModelId = watch('model_id');
  const selectedSymbols = watch('symbols');

  const modelsQuery = useQuery({
    queryKey: ['models', 'core_rl', 'backtesting-form'],
    queryFn: () => modelsApi.list({ modelType: 'core_rl', pageSize: 100 }),
  });

  const generationsQuery = useQuery({
    queryKey: ['generations', selectedModelId, 'backtesting-form'],
    queryFn: () =>
      selectedModelId
        ? trainingApi.listGenerations(selectedModelId).then((response) => response.items)
        : Promise.resolve([]),
    enabled: Boolean(selectedModelId),
  });

  const models = useMemo(() => modelsQuery.data?.items ?? [], [modelsQuery.data]);
  const generations = useMemo(
    () =>
      [...(generationsQuery.data ?? [])].sort(
        (left, right) => right.generation_number - left.generation_number,
      ),
    [generationsQuery.data],
  );

  useEffect(() => {
    if (!selectedModelId && models[0]) {
      setValue('model_id', models[0].model_id, { shouldValidate: true });
    }
  }, [models, selectedModelId, setValue]);

  useEffect(() => {
    resetField('generation_id', { defaultValue: '' });
  }, [resetField, selectedModelId]);

  useEffect(() => {
    if (selectedModelId && generations[0]) {
      setValue('generation_id', generations[0].generation_id, { shouldValidate: true });
    }
  }, [generations, selectedModelId, setValue]);

  const toggleSymbol = (symbol: string): void => {
    const normalized = normalizeSymbol(symbol);
    if (!normalized) {
      return;
    }
    const nextSymbols = selectedSymbols.includes(normalized)
      ? selectedSymbols.filter((selectedSymbol) => selectedSymbol !== normalized)
      : [...selectedSymbols, normalized];
    setValue('symbols', nextSymbols, { shouldValidate: true });
  };

  const addSymbol = (): void => {
    const normalized = normalizeSymbol(symbolInput);
    if (!normalized || selectedSymbols.includes(normalized)) {
      setSymbolInput('');
      return;
    }
    setValue('symbols', [...selectedSymbols, normalized], { shouldValidate: true });
    setSymbolInput('');
  };

  const submit = (data: BacktestFormData): void => {
    const resolvedModelId = data.model_id || selectedModelId || models[0]?.model_id || '';
    const resolvedGenerationId = data.generation_id || generations[0]?.generation_id || '';
    const resolvedStartDate = data.start_date || daysAgoIso(90);
    const resolvedEndDate = data.end_date || todayIso();
    const resolvedInitialCapital =
      Number.isFinite(data.initial_capital) && data.initial_capital > 0
        ? data.initial_capital
        : DEFAULT_INITIAL_CAPITAL;

    if (!resolvedModelId || !resolvedGenerationId) {
      return;
    }
    onSubmit({
      ...data,
      model_id: resolvedModelId,
      generation_id: resolvedGenerationId,
      start_date: resolvedStartDate,
      end_date: resolvedEndDate,
      initial_capital: resolvedInitialCapital,
    });
  };

  return (
    <form className="surface-panel overflow-hidden" onSubmit={handleSubmit(submit)}>
      <div className="border-b border-stone-200 px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="rounded-md bg-amber-50 p-2 text-caution">
            <CalendarRange size={18} aria-hidden="true" />
          </span>
          <div>
            <h3 className="font-semibold text-ink">Configure backtest</h3>
            <p className="mt-1 text-sm text-stone-600">
              Select a trained generation and historical window.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-5 p-5">
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <label htmlFor="backtest-model" className="mb-1 block text-sm font-medium text-ink">
              Model
            </label>
            <Controller
              name="model_id"
              control={control}
              render={({ field }) => (
                <select
                  id="backtest-model"
                  {...field}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
                  disabled={modelsQuery.isLoading || models.length === 0}
                >
                  {models.length === 0 && <option value="">No core RL models</option>}
                  {models.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {modelOptionLabel(model)}
                    </option>
                  ))}
                </select>
              )}
            />
            {errors.model_id && (
              <p className="mt-1 text-xs text-red-600">{errors.model_id.message}</p>
            )}
          </div>

          <div>
            <label
              htmlFor="backtest-generation"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Generation
            </label>
            <Controller
              name="generation_id"
              control={control}
              render={({ field }) => (
                <select
                  id="backtest-generation"
                  {...field}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
                  disabled={
                    !selectedModelId || generationsQuery.isLoading || generations.length === 0
                  }
                >
                  {generations.length === 0 && <option value="">No completed generations</option>}
                  {generations.map((generation) => (
                    <option key={generation.generation_id} value={generation.generation_id}>
                      {generationOptionLabel(generation)}
                    </option>
                  ))}
                </select>
              )}
            />
            {errors.generation_id && (
              <p className="mt-1 text-xs text-red-600">{errors.generation_id.message}</p>
            )}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <label
              htmlFor="backtest-start-date"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Start date
            </label>
            <Controller
              name="start_date"
              control={control}
              render={({ field }) => (
                <input
                  id="backtest-start-date"
                  type="date"
                  {...field}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
                />
              )}
            />
          </div>
          <div>
            <label htmlFor="backtest-end-date" className="mb-1 block text-sm font-medium text-ink">
              End date
            </label>
            <Controller
              name="end_date"
              control={control}
              render={({ field }) => (
                <input
                  id="backtest-end-date"
                  type="date"
                  {...field}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
                />
              )}
            />
          </div>
          <div>
            <label htmlFor="backtest-capital" className="mb-1 block text-sm font-medium text-ink">
              Initial capital
            </label>
            <Controller
              name="initial_capital"
              control={control}
              render={({ field }) => (
                <input
                  id="backtest-capital"
                  type="number"
                  min={1}
                  step={1000}
                  value={field.value}
                  onChange={(event) => field.onChange(Number(event.target.value))}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
                />
              )}
            />
            {errors.initial_capital && (
              <p className="mt-1 text-xs text-red-600">{errors.initial_capital.message}</p>
            )}
          </div>
        </div>

        <fieldset>
          <legend className="mb-2 text-sm font-medium text-ink">Symbols</legend>
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
            {DEFAULT_SYMBOLS.map((symbol) => (
              <label
                key={symbol}
                className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition ${
                  selectedSymbols.includes(symbol)
                    ? 'border-action bg-blue-50 text-ink'
                    : 'border-stone-200 bg-white text-stone-700 hover:border-stone-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedSymbols.includes(symbol)}
                  onChange={() => toggleSymbol(symbol)}
                  className="h-4 w-4 rounded border-stone-300 text-action"
                />
                {symbol}
              </label>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {selectedSymbols.map((symbol) => (
              <span
                key={symbol}
                className="inline-flex items-center gap-1 rounded-md bg-stone-100 px-2 py-1 text-sm text-ink"
              >
                {symbol}
                <button
                  type="button"
                  aria-label={`Remove ${symbol}`}
                  className="inline-flex h-5 w-5 items-center justify-center rounded-md text-stone-500 hover:bg-stone-200"
                  onClick={() => toggleSymbol(symbol)}
                >
                  <X size={13} aria-hidden="true" />
                </button>
              </span>
            ))}
          </div>
          <div className="mt-3 flex flex-col gap-2 sm:flex-row">
            <label htmlFor="backtest-symbol" className="sr-only">
              Custom symbol
            </label>
            <input
              id="backtest-symbol"
              value={symbolInput}
              onChange={(event) => setSymbolInput(event.target.value.toUpperCase())}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  addSymbol();
                }
              }}
              placeholder="Ticker"
              className="min-w-0 flex-1 rounded-md border border-stone-300 px-3 py-2 text-sm"
            />
            <button type="button" className="secondary-button" onClick={addSymbol}>
              <Plus size={16} aria-hidden="true" />
              Add
            </button>
          </div>
        </fieldset>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto]">
          <div>
            <label
              htmlFor="backtest-description"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Description
            </label>
            <Controller
              name="description"
              control={control}
              render={({ field }) => (
                <input
                  id="backtest-description"
                  {...field}
                  placeholder="Evaluation label"
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm"
                />
              )}
            />
          </div>
          <div className="flex items-end">
            <Controller
              name="include_transaction_costs"
              control={control}
              render={({ field }) => (
                <label className="flex min-h-10 items-center gap-2 rounded-md border border-stone-200 px-3 py-2 text-sm text-stone-700">
                  <input
                    type="checkbox"
                    checked={field.value}
                    onChange={(event) => field.onChange(event.target.checked)}
                    className="h-4 w-4 rounded border-stone-300 text-action"
                  />
                  Include costs
                </label>
              )}
            />
          </div>
        </div>

        {(modelsQuery.error || generationsQuery.error) && (
          <p className="text-sm text-red-600">Unable to load models or generations.</p>
        )}

        <button
          type="button"
          className="primary-button w-full sm:w-auto"
          disabled={isLoading || !selectedModelId || generations.length === 0}
          onClick={() => {
            void handleSubmit(submit)();
          }}
        >
          <Play size={16} aria-hidden="true" />
          {isLoading ? 'Running backtest' : 'Run backtest'}
        </button>
      </div>
    </form>
  );
}
