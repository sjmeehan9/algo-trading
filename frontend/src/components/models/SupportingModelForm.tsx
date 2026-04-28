import { zodResolver } from '@hookform/resolvers/zod';
import { ArrowLeft, BrainCircuit, Save, Waypoints } from 'lucide-react';
import { useEffect, useMemo } from 'react';
import { Controller, type Resolver, type SubmitHandler, useForm } from 'react-hook-form';

import { RL_ALGORITHMS } from '../../constants/algorithms';
import {
  getAlgorithmsForFramework,
  getFrameworkOptions,
  ML_ALGORITHMS,
} from '../../constants/mlAlgorithms';
import { SUPPORTING_INPUT_FREQUENCIES } from '../../constants/dataTypes';
import {
  getDefaultSupportingHyperparameters,
  getDefaultSupportingModelFormValues,
  supportingModelSchema,
  type SupportingModelCategory,
  type SupportingModelFormData,
} from '../../hooks/useSupportingModelForm';
import DataTypeSelector from './DataTypeSelector';
import HyperparameterEditor from './HyperparameterEditor';
import SignalTypeSelector from './SignalTypeSelector';

interface SupportingModelFormProps {
  initialData?: SupportingModelFormData;
  isLoading?: boolean;
  submitLabel?: string;
  errorMessage?: string | null;
  onCancel: () => void;
  onSubmit: (data: SupportingModelFormData) => Promise<void>;
}

const supportingFormResolver = zodResolver(
  supportingModelSchema,
) as unknown as Resolver<SupportingModelFormData>;

const algorithmsForCategory = (category: SupportingModelCategory) =>
  category === 'ml' ? ML_ALGORITHMS : RL_ALGORITHMS;

/** Full configuration form for ML/RL models that produce core-model signals. */
export default function SupportingModelForm({
  initialData,
  isLoading = false,
  submitLabel = 'Save Supporting Model',
  errorMessage = null,
  onCancel,
  onSubmit,
}: SupportingModelFormProps): JSX.Element {
  const startingData = useMemo(
    () => initialData ?? getDefaultSupportingModelFormValues(),
    [initialData],
  );
  const {
    control,
    handleSubmit,
    reset,
    setValue,
    watch,
    formState: { errors },
  } = useForm<SupportingModelFormData>({
    resolver: supportingFormResolver,
    defaultValues: startingData,
    mode: 'onBlur',
  });

  useEffect(() => {
    reset(startingData);
  }, [reset, startingData]);

  const modelCategory = watch('model_category');
  const selectedFramework = watch('framework');
  const selectedAlgorithm = watch('algorithm');
  const availableAlgorithms = useMemo(() => algorithmsForCategory(modelCategory), [modelCategory]);
  const frameworkOptions = useMemo(
    () => getFrameworkOptions(availableAlgorithms),
    [availableAlgorithms],
  );
  const frameworkAlgorithms = useMemo(
    () => getAlgorithmsForFramework(availableAlgorithms, selectedFramework),
    [availableAlgorithms, selectedFramework],
  );
  const algorithmDefinition =
    availableAlgorithms.find((algorithm) => algorithm.id === selectedAlgorithm) ??
    frameworkAlgorithms[0] ??
    availableAlgorithms[0]!;

  const setAlgorithmDefaults = (
    category: SupportingModelCategory,
    algorithmId: string,
    framework: string,
  ): void => {
    setValue('framework', framework, { shouldDirty: true, shouldValidate: true });
    setValue('algorithm', algorithmId, { shouldDirty: true, shouldValidate: true });
    setValue('hyperparameters', getDefaultSupportingHyperparameters(category, algorithmId), {
      shouldDirty: true,
      shouldValidate: true,
    });
  };

  const handleCategoryChange = (category: SupportingModelCategory): void => {
    const nextDefaults = getDefaultSupportingModelFormValues(category);
    setValue('model_category', category, { shouldDirty: true, shouldValidate: true });
    setAlgorithmDefaults(category, nextDefaults.algorithm, nextDefaults.framework);
  };

  const handleFrameworkChange = (framework: string): void => {
    const nextAlgorithm = getAlgorithmsForFramework(availableAlgorithms, framework)[0];
    if (!nextAlgorithm) {
      return;
    }
    setAlgorithmDefaults(modelCategory, nextAlgorithm.id, nextAlgorithm.framework);
  };

  const handleAlgorithmChange = (algorithmId: string): void => {
    const nextAlgorithm = availableAlgorithms.find((algorithm) => algorithm.id === algorithmId);
    if (!nextAlgorithm) {
      return;
    }
    setAlgorithmDefaults(modelCategory, nextAlgorithm.id, nextAlgorithm.framework);
  };

  const submitForm: SubmitHandler<SupportingModelFormData> = async (formData) => {
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
        <h3 className="text-lg font-semibold text-ink">Model Type</h3>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <button
            type="button"
            aria-pressed={modelCategory === 'ml'}
            onClick={() => handleCategoryChange('ml')}
            className={`rounded-md border px-4 py-3 text-left transition ${
              modelCategory === 'ml'
                ? 'border-action bg-blue-50 text-ink'
                : 'border-stone-200 bg-white text-stone-700 hover:border-stone-300'
            }`}
          >
            <span className="flex items-center gap-2 font-semibold">
              <BrainCircuit size={17} aria-hidden="true" />
              Machine Learning
            </span>
            <span className="mt-1 block text-sm text-stone-500">
              Supervised signal models for text, indicators, and market features.
            </span>
          </button>

          <button
            type="button"
            aria-pressed={modelCategory === 'rl'}
            onClick={() => handleCategoryChange('rl')}
            className={`rounded-md border px-4 py-3 text-left transition ${
              modelCategory === 'rl'
                ? 'border-action bg-blue-50 text-ink'
                : 'border-stone-200 bg-white text-stone-700 hover:border-stone-300'
            }`}
          >
            <span className="flex items-center gap-2 font-semibold">
              <Waypoints size={17} aria-hidden="true" />
              Reinforcement Learning
            </span>
            <span className="mt-1 block text-sm text-stone-500">
              RL models that emit supporting signals rather than live-trading decisions.
            </span>
          </button>
        </div>
      </section>

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Basic Information</h3>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <label
              htmlFor="supporting-model-name"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Model Name
            </label>
            <Controller
              name="name"
              control={control}
              render={({ field }) => (
                <input
                  {...field}
                  id="supporting-model-name"
                  placeholder="News Sentiment Signal"
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
            <label
              htmlFor="supporting-model-framework"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Framework
            </label>
            <Controller
              name="framework"
              control={control}
              render={({ field }) => (
                <select
                  id="supporting-model-framework"
                  value={field.value}
                  onChange={(event) => handleFrameworkChange(event.target.value)}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                >
                  {frameworkOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              )}
            />
            {errors.framework?.message && (
              <p className="mt-1 text-sm text-red-600">{errors.framework.message}</p>
            )}
          </div>
        </div>

        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <label
              htmlFor="supporting-model-algorithm"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Algorithm
            </label>
            <Controller
              name="algorithm"
              control={control}
              render={({ field }) => (
                <select
                  id="supporting-model-algorithm"
                  value={field.value}
                  onChange={(event) => handleAlgorithmChange(event.target.value)}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                >
                  {frameworkAlgorithms.map((algorithm) => (
                    <option key={algorithm.id} value={algorithm.id}>
                      {algorithm.name}
                    </option>
                  ))}
                </select>
              )}
            />
            <p className="mt-1 text-xs leading-5 text-stone-500">
              {algorithmDefinition.description}
            </p>
            {errors.algorithm?.message && (
              <p className="mt-1 text-sm text-red-600">{errors.algorithm.message}</p>
            )}
          </div>

          <div>
            <label
              htmlFor="supporting-model-description"
              className="mb-1 block text-sm font-medium text-ink"
            >
              Description
            </label>
            <Controller
              name="description"
              control={control}
              render={({ field }) => (
                <textarea
                  {...field}
                  value={field.value ?? ''}
                  id="supporting-model-description"
                  rows={3}
                  className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
                />
              )}
            />
            {errors.description?.message && (
              <p className="mt-1 text-sm text-red-600">{errors.description.message}</p>
            )}
          </div>
        </div>
      </section>

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Input Data</h3>
        <div className="mt-4">
          <Controller
            name="input_data_types"
            control={control}
            render={({ field }) => (
              <DataTypeSelector selected={field.value} onChange={field.onChange} />
            )}
          />
          {errors.input_data_types?.message && (
            <p className="mt-2 text-sm text-red-600">{errors.input_data_types.message}</p>
          )}
        </div>

        <div className="mt-4 max-w-xs">
          <label
            htmlFor="supporting-input-frequency"
            className="mb-1 block text-sm font-medium text-ink"
          >
            Input Frequency
          </label>
          <Controller
            name="input_frequency"
            control={control}
            render={({ field }) => (
              <select
                {...field}
                id="supporting-input-frequency"
                className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
              >
                {SUPPORTING_INPUT_FREQUENCIES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            )}
          />
          {errors.input_frequency?.message && (
            <p className="mt-1 text-sm text-red-600">{errors.input_frequency.message}</p>
          )}
        </div>
      </section>

      <section className="surface-panel p-5">
        <h3 className="text-lg font-semibold text-ink">Output Signal</h3>
        <div className="mt-4">
          <Controller
            name="signal_type"
            control={control}
            render={({ field }) => (
              <SignalTypeSelector selected={field.value} onChange={field.onChange} />
            )}
          />
          {errors.signal_type?.message && (
            <p className="mt-2 text-sm text-red-600">{errors.signal_type.message}</p>
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
