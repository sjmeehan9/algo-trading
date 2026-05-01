import { describe, expect, it } from 'vitest';

import type { ModelConfigResponse } from '../api/models';
import {
  getDefaultSupportingModelFormValues,
  modelResponseToSupportingFormData,
  supportingFormDataToCreatePayload,
  supportingModelSchema,
} from './useSupportingModelForm';

describe('supportingModelSchema', () => {
  it('accepts a complete supporting ML model configuration', () => {
    const formData = {
      ...getDefaultSupportingModelFormValues('ml'),
      name: 'News Sentiment Signal',
      input_data_types: ['news_text'],
    };

    expect(supportingModelSchema.safeParse(formData).success).toBe(true);
  });

  it('rejects supporting model signal inputs', () => {
    const formData = {
      ...getDefaultSupportingModelFormValues('ml'),
      name: 'Invalid Supporting Signal',
      input_data_types: ['model_signal'],
    };

    const result = supportingModelSchema.safeParse(formData);

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues.map((issue) => issue.message)).toContain(
        'Supporting models cannot use supporting model signals as input',
      );
    }
  });
});

describe('supporting model form payload conversion', () => {
  it('maps supporting ML form data to the backend create payload', () => {
    const formData = {
      ...getDefaultSupportingModelFormValues('ml'),
      name: ' News Sentiment Signal ',
      description: ' ',
      input_data_types: ['news_text'],
      input_frequency: 'realtime',
    };

    const payload = supportingFormDataToCreatePayload(formData);

    expect(payload).toMatchObject({
      name: 'News Sentiment Signal',
      description: null,
      model_type: 'supporting_ml',
      signal_type: 'sentiment',
      trainer_type: 'sklearn',
      algorithm: 'random_forest',
      input_data_types: ['news_text'],
      input_frequency: 'realtime',
      supporting_model_ids: [],
      strategy_ids: [],
      reward_function: null,
    });
    expect(payload.hyperparameters).toMatchObject({ n_estimators: 100, max_depth: 10 });
  });

  it('maps supporting RL form data to the backend create payload', () => {
    const formData = {
      ...getDefaultSupportingModelFormValues('rl'),
      name: 'RL Trend Signal',
      input_data_types: ['market_bar'],
      signal_type: 'trend',
    };

    const payload = supportingFormDataToCreatePayload(formData);

    expect(payload).toMatchObject({
      model_type: 'supporting_rl',
      signal_type: 'trend',
      trainer_type: 'stable_baselines3',
      algorithm: 'ppo',
      input_data_types: ['market_bar'],
    });
    expect(payload.hyperparameters).toMatchObject({ learning_rate: 0.0003 });
  });

  it('hydrates existing supporting model responses for editing', () => {
    const model: ModelConfigResponse = {
      model_id: 'supporting-ml-1',
      name: 'Existing Sentiment Model',
      description: 'Existing helper',
      model_type: 'supporting_ml',
      signal_type: 'sentiment',
      trainer_type: 'huggingface',
      algorithm: 'transformer_sentiment',
      hyperparameters: {
        model_name: 'finbert',
        max_length: 256,
      },
      training_data_config: {},
      supporting_model_ids: [],
      strategy_ids: [],
      environment_config: {},
      reward_function: null,
      input_data_types: ['news_text'],
      input_frequency: 'realtime',
      created_at: '2026-04-01T00:00:00Z',
      updated_at: '2026-04-02T00:00:00Z',
      state: 'configured',
    };

    const formData = modelResponseToSupportingFormData(model);

    expect(formData).toMatchObject({
      name: 'Existing Sentiment Model',
      model_category: 'ml',
      framework: 'huggingface',
      algorithm: 'transformer_sentiment',
      input_data_types: ['news_text'],
      input_frequency: 'realtime',
      signal_type: 'sentiment',
    });
    expect(formData.hyperparameters.max_length).toBe(256);
  });
});
