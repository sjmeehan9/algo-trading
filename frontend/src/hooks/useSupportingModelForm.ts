import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import { z } from 'zod';

import {
  DEFAULT_RL_ALGORITHM,
  DEFAULT_SESSION_END,
  DEFAULT_SESSION_START,
  DEFAULT_SESSION_TIMEZONE,
  DEFAULT_TRAINING_SYMBOLS,
  getAlgorithmDefinition,
  getDefaultHyperparameters,
  type AlgorithmDef,
  type HyperparameterValue,
} from '../constants/algorithms';
import {
  DEFAULT_ML_ALGORITHM,
  ML_ALGORITHMS,
  getDefaultMLHyperparameters,
  getMLAlgorithmDefinition,
} from '../constants/mlAlgorithms';
import {
  DEFAULT_SUPPORTING_INPUT_FREQUENCY,
  DEFAULT_SUPPORTING_SIGNAL_TYPE,
  FORBIDDEN_SUPPORTING_INPUT_TYPES,
  isSelectableSupportingInputType,
  isSupportedSignalType,
} from '../constants/dataTypes';
import {
  modelsApi,
  type ActivatePretrainedRequest,
  type HyperparameterValue as ModelHyperparameterValue,
  type LoadArtifactRequest,
  type ModelConfigCreate,
  type ModelConfigResponse,
  type ModelConfigUpdate,
  type SupportingModelLifecycleResponse,
} from '../api/models';

export type SupportingModelCategory = 'ml' | 'rl';

/**
 * Sentiment-style algorithms that map onto a concrete pretrained backend in the
 * Component 7.7 lifecycle service. Only these support `activate-pretrained`; any
 * other algorithm must be trained or have an artifact loaded instead.
 */
export const PRETRAINED_SENTIMENT_ALGORITHMS: readonly string[] = [
  'transformer_sentiment',
  'finbert',
  'vader',
  'provider',
  'provider_passthrough',
];

/** Return whether an algorithm can be activated as a pretrained sentiment backend. */
export const supportsPretrainedActivation = (algorithm: string | null | undefined): boolean =>
  Boolean(algorithm) &&
  PRETRAINED_SENTIMENT_ALGORITHMS.includes(String(algorithm).trim().toLowerCase());

const hyperparameterValueSchema = z.union([z.number(), z.string(), z.boolean()]);

const supportingCategorySchema = z.enum(['ml', 'rl']);

export const supportingModelSchema = z
  .object({
    name: z
      .string()
      .trim()
      .min(1, 'Name is required')
      .max(100, 'Name must be 100 characters or less'),
    description: z.string().max(500, 'Description must be 500 characters or less').optional(),
    model_category: supportingCategorySchema,
    framework: z.string().min(1, 'Framework is required'),
    algorithm: z.string().min(1, 'Algorithm is required'),
    hyperparameters: z.record(z.string(), hyperparameterValueSchema),
    training_data: z.object({
      symbols: z.array(z.string().trim().min(1)).min(1, 'Select at least one symbol'),
      start_date: z.string().min(1, 'Start date is required'),
      end_date: z.string().min(1, 'End date is required'),
      data_frequency: z.string().min(1, 'Data frequency is required'),
      restrict_to_session: z.boolean().default(false),
      session_start: z.string().regex(/^\d{2}:\d{2}$/, 'Use HH:MM').default(DEFAULT_SESSION_START),
      session_end: z.string().regex(/^\d{2}:\d{2}$/, 'Use HH:MM').default(DEFAULT_SESSION_END),
      session_timezone: z.string().min(1, 'Timezone is required').default(DEFAULT_SESSION_TIMEZONE),
    }),
    input_data_types: z.array(z.string()).min(1, 'Select at least one input type'),
    input_frequency: z.string().min(1, 'Input frequency is required'),
    signal_type: z.string().min(1, 'Select output signal type'),
  })
  .superRefine((value, context) => {
    const startTime = Date.parse(`${value.training_data.start_date}T00:00:00Z`);
    const endTime = Date.parse(`${value.training_data.end_date}T00:00:00Z`);
    if (Number.isFinite(startTime) && Number.isFinite(endTime) && endTime < startTime) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'End date must be on or after start date',
        path: ['training_data', 'end_date'],
      });
    }

    if (
      value.training_data.restrict_to_session &&
      value.training_data.session_end <= value.training_data.session_start
    ) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Session end must be after session start',
        path: ['training_data', 'session_end'],
      });
    }

    const algorithm = getSupportingAlgorithmDefinition(value.model_category, value.algorithm);
    const isKnownAlgorithm = getAlgorithmsForCategory(value.model_category).some(
      (candidate) => candidate.id === value.algorithm,
    );

    if (!isKnownAlgorithm) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Choose a supported algorithm for this model type',
        path: ['algorithm'],
      });
    }

    if (algorithm.framework !== value.framework) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Framework must match the selected algorithm',
        path: ['framework'],
      });
    }

    value.input_data_types.forEach((dataType, index) => {
      const normalized = dataType.trim().toLowerCase();
      if (FORBIDDEN_SUPPORTING_INPUT_TYPES.includes(normalized)) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Supporting models cannot use supporting model signals as input',
          path: ['input_data_types', index],
        });
        return;
      }

      if (!isSelectableSupportingInputType(normalized)) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Choose a supported input data type',
          path: ['input_data_types', index],
        });
      }
    });

    if (!isSupportedSignalType(value.signal_type)) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Choose a supported output signal type',
        path: ['signal_type'],
      });
    }
  });

export type SupportingModelFormData = z.infer<typeof supportingModelSchema>;

export interface UseSupportingModelFormOptions {
  enabled?: boolean;
}

export interface UseSupportingModelFormResult {
  initialData: SupportingModelFormData;
  isEditMode: boolean;
  isLoadingInitialData: boolean;
  loadError: string | null;
  saveError: string | null;
  isSaving: boolean;
  save: (formData: SupportingModelFormData) => Promise<ModelConfigResponse>;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const getStringList = (
  record: Record<string, unknown>,
  key: string,
  fallback: string[],
): string[] => {
  const value = record[key];
  if (!Array.isArray(value)) {
    return fallback;
  }

  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
};

const normalizeDescription = (description: string | null | undefined): string | null => {
  const trimmed = description?.trim() ?? '';
  return trimmed.length > 0 ? trimmed : null;
};

const toDateInputValue = (dateValue: Date): string => dateValue.toISOString().slice(0, 10);

const oneYearBefore = (dateValue: Date): Date => {
  const result = new Date(dateValue);
  result.setFullYear(result.getFullYear() - 1);
  return result;
};

const getTrainingDataString = (
  record: Record<string, unknown>,
  key: string,
  fallback: string,
): string => {
  const value = record[key];
  return typeof value === 'string' && value.trim().length > 0 ? value : fallback;
};

const getDefaultTrainingData = (): SupportingModelFormData['training_data'] => {
  const endDate = new Date();
  return {
    symbols: [DEFAULT_TRAINING_SYMBOLS[0] ?? 'SPY'],
    start_date: toDateInputValue(oneYearBefore(endDate)),
    end_date: toDateInputValue(endDate),
    data_frequency: '1m',
    restrict_to_session: false,
    session_start: DEFAULT_SESSION_START,
    session_end: DEFAULT_SESSION_END,
    session_timezone: DEFAULT_SESSION_TIMEZONE,
  };
};

const trainingDataFromConfig = (
  trainingConfig: Record<string, unknown>,
): SupportingModelFormData['training_data'] => {
  const defaults = getDefaultTrainingData();
  const symbols = getStringList(trainingConfig, 'symbols', defaults.symbols);
  return {
    symbols: symbols.length > 0 ? symbols : defaults.symbols,
    start_date: getTrainingDataString(trainingConfig, 'start_date', defaults.start_date),
    end_date: getTrainingDataString(trainingConfig, 'end_date', defaults.end_date),
    data_frequency: getTrainingDataString(
      trainingConfig,
      'data_frequency',
      defaults.data_frequency,
    ),
    restrict_to_session: Boolean(trainingConfig.session_start && trainingConfig.session_end),
    session_start: getTrainingDataString(trainingConfig, 'session_start', defaults.session_start),
    session_end: getTrainingDataString(trainingConfig, 'session_end', defaults.session_end),
    session_timezone: getTrainingDataString(
      trainingConfig,
      'session_timezone',
      defaults.session_timezone,
    ),
  };
};

const getAlgorithmsForCategory = (category: SupportingModelCategory): AlgorithmDef[] =>
  category === 'ml'
    ? ML_ALGORITHMS
    : [getAlgorithmDefinition('ppo'), getAlgorithmDefinition('dqn'), getAlgorithmDefinition('a2c')];

/** Return the configured algorithm definition for a supporting model category. */
export const getSupportingAlgorithmDefinition = (
  category: SupportingModelCategory,
  algorithmId: string,
): AlgorithmDef =>
  category === 'ml' ? getMLAlgorithmDefinition(algorithmId) : getAlgorithmDefinition(algorithmId);

/** Build default hyperparameters for a supporting model algorithm. */
export const getDefaultSupportingHyperparameters = (
  category: SupportingModelCategory,
  algorithmId?: string,
): Record<string, HyperparameterValue> =>
  category === 'ml'
    ? getDefaultMLHyperparameters(algorithmId ?? DEFAULT_ML_ALGORITHM)
    : getDefaultHyperparameters(algorithmId ?? DEFAULT_RL_ALGORITHM);

const cleanHyperparameters = (
  category: SupportingModelCategory,
  algorithmId: string,
  hyperparameters: Record<string, HyperparameterValue>,
): Record<string, HyperparameterValue> => ({
  ...getDefaultSupportingHyperparameters(category, algorithmId),
  ...hyperparameters,
});

/** Build default form values for creating a supporting model. */
export const getDefaultSupportingModelFormValues = (
  category: SupportingModelCategory = 'ml',
): SupportingModelFormData => {
  const algorithm =
    category === 'ml'
      ? getMLAlgorithmDefinition(DEFAULT_ML_ALGORITHM)
      : getAlgorithmDefinition(DEFAULT_RL_ALGORITHM);

  return {
    name: '',
    description: '',
    model_category: category,
    framework: algorithm.framework,
    algorithm: algorithm.id,
    hyperparameters: getDefaultSupportingHyperparameters(category, algorithm.id),
    training_data: getDefaultTrainingData(),
    input_data_types: [],
    input_frequency: DEFAULT_SUPPORTING_INPUT_FREQUENCY,
    signal_type: DEFAULT_SUPPORTING_SIGNAL_TYPE,
  };
};

/** Convert a backend supporting model response into editable form data. */
export const modelResponseToSupportingFormData = (
  model: ModelConfigResponse,
): SupportingModelFormData => {
  const category: SupportingModelCategory = model.model_type === 'supporting_rl' ? 'rl' : 'ml';
  const defaults = getDefaultSupportingModelFormValues(category);
  const algorithm = getSupportingAlgorithmDefinition(
    category,
    model.algorithm || defaults.algorithm,
  );
  const configRecord = isRecord(model) ? (model as unknown as Record<string, unknown>) : {};
  const inputDataTypes = getStringList(
    configRecord,
    'input_data_types',
    defaults.input_data_types,
  ).filter(isSelectableSupportingInputType);
  const trainingConfig = isRecord(model.training_data_config) ? model.training_data_config : {};

  return {
    name: model.name,
    description: model.description ?? '',
    model_category: category,
    framework: algorithm.framework,
    algorithm: algorithm.id,
    hyperparameters: cleanHyperparameters(category, algorithm.id, model.hyperparameters),
    training_data: trainingDataFromConfig(trainingConfig),
    input_data_types: inputDataTypes,
    input_frequency: model.input_frequency ?? defaults.input_frequency,
    signal_type: model.signal_type ?? defaults.signal_type,
  };
};

/** Convert validated supporting model form data into a backend create payload. */
export const supportingFormDataToCreatePayload = (
  formData: SupportingModelFormData,
): ModelConfigCreate => {
  const algorithm = getSupportingAlgorithmDefinition(formData.model_category, formData.algorithm);
  return {
    name: formData.name.trim(),
    description: normalizeDescription(formData.description),
    model_type: formData.model_category === 'ml' ? 'supporting_ml' : 'supporting_rl',
    signal_type: formData.signal_type as ModelConfigCreate['signal_type'],
    trainer_type: algorithm.framework,
    algorithm: algorithm.id,
    hyperparameters: cleanHyperparameters(
      formData.model_category,
      algorithm.id,
      formData.hyperparameters,
    ),
    training_data_config: {
      symbols: formData.training_data.symbols.map((symbol) => symbol.trim().toUpperCase()),
      start_date: formData.training_data.start_date,
      end_date: formData.training_data.end_date,
      data_frequency: formData.training_data.data_frequency,
      // Only send the session window when explicitly enabled; omitting these
      // keys leaves the backend on full-day (useRTH) acquisition.
      ...(formData.training_data.restrict_to_session
        ? {
            session_start: formData.training_data.session_start,
            session_end: formData.training_data.session_end,
            session_timezone: formData.training_data.session_timezone,
          }
        : {}),
    },
    supporting_model_ids: [],
    strategy_ids: [],
    environment_config: {},
    reward_function: null,
    input_data_types: [...formData.input_data_types],
    input_frequency: formData.input_frequency,
  };
};

/** Convert validated supporting model form data into a backend update payload. */
export const supportingFormDataToUpdatePayload = (
  formData: SupportingModelFormData,
): ModelConfigUpdate => supportingFormDataToCreatePayload(formData);

const errorToMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to save supporting model configuration.';

/** Manage loading and saving state for supporting model configuration pages. */
export const useSupportingModelForm = (
  modelId?: string,
  options: UseSupportingModelFormOptions = {},
): UseSupportingModelFormResult => {
  const queryClient = useQueryClient();
  const enabled = options.enabled ?? true;
  const isEditMode = Boolean(modelId);

  const modelQuery = useQuery({
    queryKey: ['models', modelId],
    queryFn: () => modelsApi.get(modelId ?? ''),
    enabled: enabled && isEditMode,
  });

  const createMutation = useMutation({
    mutationFn: (formData: SupportingModelFormData) =>
      modelsApi.create(supportingFormDataToCreatePayload(formData)),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['models'] }),
        // Surface the new supporting model in the core RL input selector.
        queryClient.invalidateQueries({ queryKey: ['core-rl-input-options'] }),
      ]);
    },
  });

  const updateMutation = useMutation({
    mutationFn: (formData: SupportingModelFormData) => {
      if (!modelId) {
        throw new Error('Model ID is required for updates.');
      }
      return modelsApi.update(modelId, supportingFormDataToUpdatePayload(formData));
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['models'] }),
        queryClient.invalidateQueries({ queryKey: ['models', modelId] }),
        queryClient.invalidateQueries({ queryKey: ['core-rl-input-options'] }),
      ]);
    },
  });

  const unsupportedModelType = modelQuery.data?.model_type === 'core_rl';
  const initialData = useMemo(
    () =>
      modelQuery.data && !unsupportedModelType
        ? modelResponseToSupportingFormData(modelQuery.data)
        : getDefaultSupportingModelFormValues(),
    [modelQuery.data, unsupportedModelType],
  );

  const loadError = unsupportedModelType
    ? 'This configuration is not a supporting model.'
    : modelQuery.error
      ? errorToMessage(modelQuery.error)
      : null;
  const saveError = createMutation.error
    ? errorToMessage(createMutation.error)
    : updateMutation.error
      ? errorToMessage(updateMutation.error)
      : null;

  return {
    initialData,
    isEditMode,
    isLoadingInitialData: modelQuery.isLoading,
    loadError,
    saveError,
    isSaving: createMutation.isPending || updateMutation.isPending,
    save: (formData) =>
      isEditMode ? updateMutation.mutateAsync(formData) : createMutation.mutateAsync(formData),
  };
};

/** React Query key for one supporting model's lifecycle snapshot. */
export const supportingLifecycleQueryKey = (modelId: string): readonly unknown[] => [
  'models',
  modelId,
  'lifecycle',
];

export interface UseSupportingModelLifecycleResult {
  lifecycle: SupportingModelLifecycleResponse | undefined;
  isLoading: boolean;
  loadError: string | null;
  actionError: string | null;
  isMutating: boolean;
  activatePretrained: (
    overrides?: Record<string, ModelHyperparameterValue>,
  ) => Promise<SupportingModelLifecycleResponse>;
  loadArtifact: (modelPath: string) => Promise<SupportingModelLifecycleResponse>;
  unload: () => Promise<SupportingModelLifecycleResponse>;
  refetch: () => void;
}

const lifecycleErrorToMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Lifecycle operation failed.';

/**
 * Manage the lifecycle query and the three readiness actions (activate
 * pretrained, load artifact, unload) for a single supporting model.
 *
 * On every successful action the supporting registry state changes server-side,
 * so this hook invalidates the model list, the single-model query, the lifecycle
 * query, and the core RL input-option query. That last invalidation lets a core
 * RL configuration pick up a newly-ready supporting input without a page reload.
 */
export const useSupportingModelLifecycle = (
  modelId: string | undefined,
  options: { enabled?: boolean } = {},
): UseSupportingModelLifecycleResult => {
  const queryClient = useQueryClient();
  const enabled = (options.enabled ?? true) && Boolean(modelId);

  const lifecycleQuery = useQuery({
    queryKey: modelId ? supportingLifecycleQueryKey(modelId) : ['models', 'lifecycle', 'disabled'],
    queryFn: () => modelsApi.getLifecycle(modelId ?? ''),
    enabled,
  });

  const invalidateAfterAction = async (
    snapshot: SupportingModelLifecycleResponse,
  ): Promise<void> => {
    if (!modelId) {
      return;
    }
    // Seed the lifecycle cache with the authoritative backend snapshot so the
    // panel reflects the returned state rather than an optimistic value.
    queryClient.setQueryData(supportingLifecycleQueryKey(modelId), snapshot);
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['models'] }),
      queryClient.invalidateQueries({ queryKey: ['models', modelId] }),
      queryClient.invalidateQueries({ queryKey: supportingLifecycleQueryKey(modelId) }),
      // Refresh the core RL supporting-input selector so a newly-ready model is
      // immediately selectable without a manual page reload.
      queryClient.invalidateQueries({ queryKey: ['core-rl-input-options'] }),
    ]);
  };

  const activateMutation = useMutation({
    mutationFn: (overrides?: Record<string, ModelHyperparameterValue>) => {
      if (!modelId) {
        throw new Error('Model ID is required to activate a pretrained model.');
      }
      const request: ActivatePretrainedRequest =
        overrides && Object.keys(overrides).length > 0 ? { hyperparameters: overrides } : {};
      return modelsApi.activatePretrained(modelId, request);
    },
    onSuccess: invalidateAfterAction,
  });

  const loadArtifactMutation = useMutation({
    mutationFn: (modelPath: string) => {
      if (!modelId) {
        throw new Error('Model ID is required to load an artifact.');
      }
      const request: LoadArtifactRequest = { model_path: modelPath };
      return modelsApi.loadArtifact(modelId, request);
    },
    onSuccess: invalidateAfterAction,
  });

  const unloadMutation = useMutation({
    mutationFn: () => {
      if (!modelId) {
        throw new Error('Model ID is required to unload a model.');
      }
      return modelsApi.unload(modelId);
    },
    onSuccess: invalidateAfterAction,
  });

  const actionError =
    activateMutation.error || loadArtifactMutation.error || unloadMutation.error
      ? lifecycleErrorToMessage(
          activateMutation.error || loadArtifactMutation.error || unloadMutation.error,
        )
      : null;

  return {
    lifecycle: lifecycleQuery.data,
    isLoading: lifecycleQuery.isLoading,
    loadError: lifecycleQuery.error ? lifecycleErrorToMessage(lifecycleQuery.error) : null,
    actionError,
    isMutating:
      activateMutation.isPending || loadArtifactMutation.isPending || unloadMutation.isPending,
    activatePretrained: (overrides) => activateMutation.mutateAsync(overrides),
    loadArtifact: (modelPath) => loadArtifactMutation.mutateAsync(modelPath),
    unload: () => unloadMutation.mutateAsync(),
    refetch: () => {
      void lifecycleQuery.refetch();
    },
  };
};
