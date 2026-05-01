import { zodResolver } from '@hookform/resolvers/zod';
import { ArrowLeft, Save } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';
import { Controller, type Resolver, type SubmitHandler, useForm } from 'react-hook-form';

import {
  REWARD_FUNCTIONS,
  RL_ALGORITHMS,
  getAlgorithmDefinition,
  getDefaultHyperparameters,
} from '../../constants/algorithms';
import {
  coreRLModelSchema,
  getDefaultCoreRLFormValues,
  type CoreRLModelFormData,
} from '../../hooks/useModelForm';
import DataSelector from './DataSelector';
import EnvironmentConfig from './EnvironmentConfig';
import HyperparameterEditor from './HyperparameterEditor';
import InputSelector from './InputSelector';

interface CoreRLModelFormProps {
  initialData?: CoreRLModelFormData;
  isLoading?: boolean;
  submitLabel?: string;
  errorMessage?: string | null;
  onCancel: () => void;
  onSubmit: (data: CoreRLModelFormData) => Promise<void>;
}

const coreRLFormResolver = zodResolver(
  coreRLModelSchema,
) as unknown as Resolver<CoreRLModelFormData>;

/** Full configuration form for core reinforcement-learning trading models. */
export default function CoreRLModelForm({
  initialData,
  isLoading = false,
  submitLabel = 'Save Model',
  errorMessage = null,
  onCancel,
  onSubmit,
}: CoreRLModelFormProps): JSX.Element {
  const startingData = useMemo(() => initialData ?? getDefaultCoreRLFormValues(), [initialData]);
  const previousAlgorithmRef = useRef(startingData.algorithm);
  const {
    control,
    handleSubmit,
    reset,
    setValue,
    watch,
    formState: { errors },
  } = useForm<CoreRLModelFormData>({
    resolver: coreRLFormResolver,
    defaultValues: startingData,
    mode: 'onBlur',
  });

  useEffect(() => {
    reset(startingData);
    previousAlgorithmRef.current = startingData.algorithm;
  }, [reset, startingData]);

  const selectedAlgorithm = watch('algorithm');
  const algorithmDefinition = useMemo(
    () => getAlgorithmDefinition(selectedAlgorithm),
    [selectedAlgorithm],
  );

  useEffect(() => {
    if (previousAlgorithmRef.current === selectedAlgorithm) {
      return;
    }

    setValue('hyperparameters', getDefaultHyperparameters(selectedAlgorithm), {
      shouldDirty: true,
      shouldValidate: true,
    });
    previousAlgorithmRef.current = selectedAlgorithm;
  }, [selectedAlgorithm, setValue]);

  const submitForm: SubmitHandler<CoreRLModelFormData> = async (formData) => {
    await onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit(submitForm)} className="space-y-6" noValidate>
      {errorMessage && (
        <div
          className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          role="alert"
        >
          {errorMessage}
        </div>
      )}

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Basic Information</h3>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <label htmlFor="model-name" className="mb-1 block text-sm font-medium text-ink">
              Model Name
            </label>
            <Controller
              name="name"
              control={control}
              render={({ field }) => (
                <input
                  {...field}
                  id="model-name"
                  placeholder="Intraday PPO Core"
                  aria-invalid={Boolean(errors.name)}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                />
              )}
            />
            {errors.name?.message && (
              <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
            )}
          </div>

          <div>
            <label htmlFor="model-algorithm" className="mb-1 block text-sm font-medium text-ink">
              Algorithm
            </label>
            <Controller
              name="algorithm"
              control={control}
              render={({ field }) => (
                <select
                  {...field}
                  id="model-algorithm"
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                >
                  {RL_ALGORITHMS.map((algorithm) => {
                    return (
                      <option key={algorithm.id} value={algorithm.id}>
                        {algorithm.name}
                      </option>
                    );
                  })}
                </select>
              )}
            />
            <p className="mt-1 text-xs leading-5 text-stone-500">
              {algorithmDefinition.description}
            </p>
          </div>
        </div>

        <div className="mt-4">
          <label htmlFor="model-description" className="mb-1 block text-sm font-medium text-ink">
            Description
          </label>
          <Controller
            name="description"
            control={control}
            render={({ field }) => (
              <textarea
                {...field}
                value={field.value ?? ''}
                id="model-description"
                rows={3}
                className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
              />
            )}
          />
          {errors.description?.message && (
            <p className="mt-1 text-sm text-red-600">{errors.description.message}</p>
          )}
        </div>
      </section>

      <section className="surface-panel p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-lg font-semibold text-ink">Hyperparameters</h3>
          <span className="text-sm text-stone-500">{algorithmDefinition.framework}</span>
        </div>
        <div className="mt-4">
          <Controller
            name="hyperparameters"
            control={control}
            render={({ field }) => (
              <HyperparameterEditor
                definitions={algorithmDefinition.hyperparameters}
                values={field.value}
                onChange={field.onChange}
              />
            )}
          />
        </div>
      </section>

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Training Data</h3>
        <div className="mt-4">
          <Controller
            name="training_data"
            control={control}
            render={({ field }) => <DataSelector value={field.value} onChange={field.onChange} />}
          />
          {errors.training_data?.symbols?.message && (
            <p className="mt-2 text-sm text-red-600">{errors.training_data.symbols.message}</p>
          )}
          {errors.training_data?.end_date?.message && (
            <p className="mt-2 text-sm text-red-600">{errors.training_data.end_date.message}</p>
          )}
        </div>
      </section>

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Signal Inputs</h3>
        <div className="mt-4 space-y-5">
          <div>
            <h4 className="mb-2 text-sm font-semibold text-ink">Supporting Models</h4>
            <Controller
              name="supporting_model_ids"
              control={control}
              render={({ field }) => (
                <InputSelector type="models" selected={field.value} onChange={field.onChange} />
              )}
            />
          </div>

          <div>
            <h4 className="mb-2 text-sm font-semibold text-ink">Strategies</h4>
            <Controller
              name="strategy_ids"
              control={control}
              render={({ field }) => (
                <InputSelector type="strategies" selected={field.value} onChange={field.onChange} />
              )}
            />
          </div>
        </div>
      </section>

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Environment</h3>
        <div className="mt-4">
          <Controller
            name="environment_config"
            control={control}
            render={({ field }) => (
              <EnvironmentConfig value={field.value} onChange={field.onChange} />
            )}
          />
        </div>
      </section>

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Training Configuration</h3>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <label htmlFor="reward-function" className="mb-1 block text-sm font-medium text-ink">
              Reward Function
            </label>
            <Controller
              name="reward_function"
              control={control}
              render={({ field }) => (
                <select
                  {...field}
                  id="reward-function"
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                >
                  {REWARD_FUNCTIONS.map((rewardFunction) => (
                    <option key={rewardFunction.id} value={rewardFunction.id}>
                      {rewardFunction.name}
                    </option>
                  ))}
                </select>
              )}
            />
          </div>

          <div>
            <label htmlFor="total-timesteps" className="mb-1 block text-sm font-medium text-ink">
              Total Timesteps
            </label>
            <Controller
              name="total_timesteps"
              control={control}
              render={({ field }) => (
                <input
                  id="total-timesteps"
                  type="number"
                  min={1_000}
                  max={10_000_000}
                  step={10_000}
                  value={field.value}
                  onChange={(event) => field.onChange(Number(event.target.value))}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                />
              )}
            />
            {errors.total_timesteps?.message && (
              <p className="mt-1 text-sm text-red-600">{errors.total_timesteps.message}</p>
            )}
          </div>
        </div>
      </section>

      <div className="flex flex-wrap justify-end gap-3">
        <button type="button" onClick={onCancel} className="secondary-button">
          <ArrowLeft size={16} aria-hidden="true" />
          Cancel
        </button>
        <button type="submit" disabled={isLoading} className="primary-button">
          <Save size={16} aria-hidden="true" />
          {isLoading ? 'Saving' : submitLabel}
        </button>
      </div>
    </form>
  );
}
