import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import { z } from 'zod';

import {
  DEFAULT_RL_ALGORITHM,
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
  type ModelConfigCreate,
  type ModelConfigResponse,
  type ModelConfigUpdate,
} from '../api/models';

export type SupportingModelCategory = 'ml' | 'rl';

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
    input_data_types: z.array(z.string()).min(1, 'Select at least one input type'),
    input_frequency: z.string().min(1, 'Input frequency is required'),
    signal_type: z.string().min(1, 'Select output signal type'),
  })
  .superRefine((value, context) => {
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

  return {
    name: model.name,
    description: model.description ?? '',
    model_category: category,
    framework: algorithm.framework,
    algorithm: algorithm.id,
    hyperparameters: cleanHyperparameters(category, algorithm.id, model.hyperparameters),
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
    training_data_config: {},
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
      await queryClient.invalidateQueries({ queryKey: ['models'] });
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
      await queryClient.invalidateQueries({ queryKey: ['models'] });
      await queryClient.invalidateQueries({ queryKey: ['models', modelId] });
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
