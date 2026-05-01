import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import { z } from 'zod';

import {
  DEFAULT_ENVIRONMENT_CONFIG,
  DEFAULT_RL_ALGORITHM,
  DEFAULT_TRAINING_SYMBOLS,
  getAlgorithmDefinition,
  getDefaultHyperparameters,
  type HyperparameterValue,
} from '../constants/algorithms';
import {
  modelsApi,
  type ModelConfigCreate,
  type ModelConfigResponse,
  type ModelConfigUpdate,
} from '../api/models';

const hyperparameterValueSchema = z.union([z.number(), z.string(), z.boolean()]);

const environmentConfigSchema = z.object({
  action_space: z.string().min(1, 'Action space is required'),
  observation_window: z.coerce.number().int().min(1).max(10_000),
  initial_cash: z.coerce.number().positive().max(1_000_000_000),
  transaction_cost_bps: z.coerce.number().min(0).max(1_000),
  max_position_size: z.coerce.number().positive().max(1),
  allow_short_selling: z.boolean(),
});

export const coreRLModelSchema = z
  .object({
    name: z
      .string()
      .trim()
      .min(1, 'Name is required')
      .max(100, 'Name must be 100 characters or less'),
    description: z.string().max(500, 'Description must be 500 characters or less').optional(),
    algorithm: z.string().min(1, 'Algorithm is required'),
    hyperparameters: z.record(z.string(), hyperparameterValueSchema),
    training_data: z.object({
      symbols: z.array(z.string().trim().min(1)).min(1, 'Select at least one symbol'),
      start_date: z.string().min(1, 'Start date is required'),
      end_date: z.string().min(1, 'End date is required'),
      data_frequency: z.string().min(1, 'Data frequency is required'),
    }),
    supporting_model_ids: z.array(z.string()),
    strategy_ids: z.array(z.string()),
    reward_function: z.string().min(1, 'Select a reward function'),
    total_timesteps: z.coerce.number().int().min(1_000).max(10_000_000),
    environment_config: environmentConfigSchema,
  })
  .superRefine((value, context) => {
    const supportedAlgorithm = getAlgorithmDefinition(value.algorithm);
    if (supportedAlgorithm.id !== value.algorithm) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Choose a supported RL algorithm',
        path: ['algorithm'],
      });
    }

    const startTime = Date.parse(`${value.training_data.start_date}T00:00:00Z`);
    const endTime = Date.parse(`${value.training_data.end_date}T00:00:00Z`);
    if (Number.isFinite(startTime) && Number.isFinite(endTime) && endTime < startTime) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'End date must be on or after start date',
        path: ['training_data', 'end_date'],
      });
    }
  });

export type CoreRLModelFormData = z.infer<typeof coreRLModelSchema>;

export interface UseModelFormResult {
  initialData: CoreRLModelFormData;
  isEditMode: boolean;
  isLoadingInitialData: boolean;
  loadError: string | null;
  saveError: string | null;
  isSaving: boolean;
  save: (formData: CoreRLModelFormData) => Promise<ModelConfigResponse>;
}

export interface UseModelFormOptions {
  enabled?: boolean;
}

const toDateInputValue = (dateValue: Date): string => dateValue.toISOString().slice(0, 10);

const today = (): Date => new Date();

const oneYearBefore = (dateValue: Date): Date => {
  const result = new Date(dateValue);
  result.setFullYear(result.getFullYear() - 1);
  return result;
};

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

  const strings = value.filter(
    (item): item is string => typeof item === 'string' && item.trim().length > 0,
  );
  return strings.length > 0 ? strings : fallback;
};

const getString = (record: Record<string, unknown>, key: string, fallback: string): string => {
  const value = record[key];
  return typeof value === 'string' && value.trim().length > 0 ? value : fallback;
};

const getNumber = (record: Record<string, unknown>, key: string, fallback: number): number => {
  const value = record[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
};

const getBoolean = (record: Record<string, unknown>, key: string, fallback: boolean): boolean => {
  const value = record[key];
  return typeof value === 'boolean' ? value : fallback;
};

const normalizeDescription = (description: string | null | undefined): string | null => {
  const trimmed = description?.trim() ?? '';
  return trimmed.length > 0 ? trimmed : null;
};

const cleanHyperparameters = (
  algorithm: string,
  hyperparameters: Record<string, HyperparameterValue>,
): Record<string, HyperparameterValue> => {
  const defaults = getDefaultHyperparameters(algorithm);
  const cleaned = Object.fromEntries(
    Object.entries(hyperparameters).filter(([key]) => key !== 'total_timesteps'),
  );
  return { ...defaults, ...cleaned };
};

/** Build default form values for creating a core RL model. */
export const getDefaultCoreRLFormValues = (): CoreRLModelFormData => {
  const endDate = today();
  const startDate = oneYearBefore(endDate);

  return {
    name: '',
    description: '',
    algorithm: DEFAULT_RL_ALGORITHM,
    hyperparameters: getDefaultHyperparameters(DEFAULT_RL_ALGORITHM),
    training_data: {
      symbols: [DEFAULT_TRAINING_SYMBOLS[0] ?? 'SPY'],
      start_date: toDateInputValue(startDate),
      end_date: toDateInputValue(endDate),
      data_frequency: '1m',
    },
    supporting_model_ids: [],
    strategy_ids: [],
    reward_function: 'profit_seeker',
    total_timesteps: 100_000,
    environment_config: { ...DEFAULT_ENVIRONMENT_CONFIG },
  };
};

/** Convert a backend core RL model response into editable form data. */
export const modelResponseToCoreRLFormData = (model: ModelConfigResponse): CoreRLModelFormData => {
  const defaults = getDefaultCoreRLFormValues();
  const trainingData = isRecord(model.training_data_config) ? model.training_data_config : {};
  const environmentConfig = isRecord(model.environment_config) ? model.environment_config : {};
  const totalTimesteps =
    typeof model.hyperparameters.total_timesteps === 'number'
      ? model.hyperparameters.total_timesteps
      : defaults.total_timesteps;

  return {
    name: model.name,
    description: model.description ?? '',
    algorithm: model.algorithm || DEFAULT_RL_ALGORITHM,
    hyperparameters: cleanHyperparameters(
      model.algorithm || DEFAULT_RL_ALGORITHM,
      model.hyperparameters,
    ),
    training_data: {
      symbols: getStringList(trainingData, 'symbols', defaults.training_data.symbols),
      start_date: getString(trainingData, 'start_date', defaults.training_data.start_date),
      end_date: getString(trainingData, 'end_date', defaults.training_data.end_date),
      data_frequency: getString(
        trainingData,
        'data_frequency',
        defaults.training_data.data_frequency,
      ),
    },
    supporting_model_ids: [...(model.supporting_model_ids ?? [])],
    strategy_ids: [...(model.strategy_ids ?? [])],
    reward_function: model.reward_function ?? defaults.reward_function,
    total_timesteps: totalTimesteps,
    environment_config: {
      action_space: getString(
        environmentConfig,
        'action_space',
        defaults.environment_config.action_space,
      ),
      observation_window: getNumber(
        environmentConfig,
        'observation_window',
        defaults.environment_config.observation_window,
      ),
      initial_cash: getNumber(
        environmentConfig,
        'initial_cash',
        defaults.environment_config.initial_cash,
      ),
      transaction_cost_bps: getNumber(
        environmentConfig,
        'transaction_cost_bps',
        defaults.environment_config.transaction_cost_bps,
      ),
      max_position_size: getNumber(
        environmentConfig,
        'max_position_size',
        defaults.environment_config.max_position_size,
      ),
      allow_short_selling: getBoolean(
        environmentConfig,
        'allow_short_selling',
        defaults.environment_config.allow_short_selling,
      ),
    },
  };
};

/** Convert validated form data into the backend create payload. */
export const coreRLFormDataToCreatePayload = (formData: CoreRLModelFormData): ModelConfigCreate => {
  const algorithm = getAlgorithmDefinition(formData.algorithm);
  return {
    name: formData.name.trim(),
    description: normalizeDescription(formData.description),
    model_type: 'core_rl',
    signal_type: null,
    trainer_type: algorithm.framework,
    algorithm: algorithm.id,
    hyperparameters: {
      ...cleanHyperparameters(algorithm.id, formData.hyperparameters),
      total_timesteps: formData.total_timesteps,
    },
    training_data_config: {
      symbols: formData.training_data.symbols.map((symbol) => symbol.trim().toUpperCase()),
      start_date: formData.training_data.start_date,
      end_date: formData.training_data.end_date,
      data_frequency: formData.training_data.data_frequency,
    },
    supporting_model_ids: [...formData.supporting_model_ids],
    strategy_ids: [...formData.strategy_ids],
    environment_config: { ...formData.environment_config },
    reward_function: formData.reward_function,
    input_data_types: [],
    input_frequency: null,
  };
};

/** Convert validated form data into the backend update payload. */
export const coreRLFormDataToUpdatePayload = (formData: CoreRLModelFormData): ModelConfigUpdate =>
  coreRLFormDataToCreatePayload(formData);

const errorToMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to save model configuration.';

/** Manage loading and saving state for core RL model configuration pages. */
export const useModelForm = (
  modelId?: string,
  options: UseModelFormOptions = {},
): UseModelFormResult => {
  const queryClient = useQueryClient();
  const enabled = options.enabled ?? true;
  const isEditMode = Boolean(modelId);

  const modelQuery = useQuery({
    queryKey: ['models', modelId],
    queryFn: () => modelsApi.get(modelId ?? ''),
    enabled: enabled && isEditMode,
  });

  const createMutation = useMutation({
    mutationFn: (formData: CoreRLModelFormData) =>
      modelsApi.create(coreRLFormDataToCreatePayload(formData)),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });

  const updateMutation = useMutation({
    mutationFn: (formData: CoreRLModelFormData) => {
      if (!modelId) {
        throw new Error('Model ID is required for updates.');
      }
      return modelsApi.update(modelId, coreRLFormDataToUpdatePayload(formData));
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['models'] });
      await queryClient.invalidateQueries({ queryKey: ['models', modelId] });
    },
  });

  const unsupportedModelType =
    modelQuery.data?.model_type && modelQuery.data.model_type !== 'core_rl';
  const initialData = useMemo(
    () =>
      modelQuery.data && !unsupportedModelType
        ? modelResponseToCoreRLFormData(modelQuery.data)
        : getDefaultCoreRLFormValues(),
    [modelQuery.data, unsupportedModelType],
  );

  const loadError = unsupportedModelType
    ? 'This configuration is not a core RL model.'
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
