import { describe, expect, it } from 'vitest';

import type { ModelConfigResponse } from '../api/models';
import {
  coreRLFormDataToCreatePayload,
  coreRLModelSchema,
  getDefaultCoreRLFormValues,
  modelResponseToCoreRLFormData,
} from './useModelForm';

describe('coreRLModelSchema', () => {
  it('accepts a complete core RL model configuration', () => {
    const formData = {
      ...getDefaultCoreRLFormValues(),
      name: 'Intraday PPO Core',
    };

    expect(coreRLModelSchema.safeParse(formData).success).toBe(true);
  });

  it('rejects missing symbols and reversed date ranges', () => {
    const formData = {
      ...getDefaultCoreRLFormValues(),
      name: 'Invalid Date Range',
      training_data: {
        symbols: [],
        start_date: '2026-04-02',
        end_date: '2026-04-01',
        data_frequency: '1m',
      },
    };

    const result = coreRLModelSchema.safeParse(formData);

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues.map((issue) => issue.message)).toEqual(
        expect.arrayContaining([
          'Select at least one symbol',
          'End date must be on or after start date',
        ]),
      );
    }
  });
});

describe('core RL form payload conversion', () => {
  it('maps validated form data to the backend create payload', () => {
    const formData = {
      ...getDefaultCoreRLFormValues(),
      name: ' Core PPO ',
      description: ' ',
      supporting_model_ids: ['supporting-ml-1'],
      strategy_ids: ['strategy-1'],
      total_timesteps: 250_000,
    };

    const payload = coreRLFormDataToCreatePayload(formData);

    expect(payload).toMatchObject({
      name: 'Core PPO',
      description: null,
      model_type: 'core_rl',
      signal_type: null,
      trainer_type: 'stable_baselines3',
      algorithm: 'ppo',
      supporting_model_ids: ['supporting-ml-1'],
      strategy_ids: ['strategy-1'],
      reward_function: 'profit_seeker',
    });
    expect(payload.hyperparameters).toMatchObject({
      learning_rate: 0.0003,
      total_timesteps: 250_000,
    });
    expect(payload.training_data_config).toMatchObject({ symbols: ['SPY'], data_frequency: '1m' });
  });

  it('hydrates existing core RL model responses for editing', () => {
    const model: ModelConfigResponse = {
      model_id: 'core-rl-1',
      name: 'Existing DQN',
      description: 'Live candidate',
      model_type: 'core_rl',
      signal_type: null,
      trainer_type: 'stable_baselines3',
      algorithm: 'dqn',
      hyperparameters: {
        learning_rate: 0.0001,
        buffer_size: 500_000,
        batch_size: 64,
        gamma: 0.98,
        exploration_fraction: 0.2,
        target_update_interval: 5_000,
        total_timesteps: 300_000,
      },
      training_data_config: {
        symbols: ['MSFT', 'NVDA'],
        start_date: '2025-01-01',
        end_date: '2025-06-30',
        data_frequency: '5m',
      },
      supporting_model_ids: ['supporting-ml-1'],
      strategy_ids: ['strategy-1'],
      environment_config: {
        action_space: 'continuous',
        observation_window: 120,
        initial_cash: 250_000,
        transaction_cost_bps: 2,
        max_position_size: 0.5,
        allow_short_selling: true,
      },
      reward_function: 'risk_adjusted',
      input_data_types: [],
      input_frequency: null,
      created_at: '2026-04-01T00:00:00Z',
      updated_at: '2026-04-02T00:00:00Z',
      state: 'configured',
    };

    const formData = modelResponseToCoreRLFormData(model);

    expect(formData).toMatchObject({
      name: 'Existing DQN',
      algorithm: 'dqn',
      supporting_model_ids: ['supporting-ml-1'],
      strategy_ids: ['strategy-1'],
      reward_function: 'risk_adjusted',
      total_timesteps: 300_000,
    });
    expect(formData.training_data.symbols).toEqual(['MSFT', 'NVDA']);
    expect(formData.environment_config.allow_short_selling).toBe(true);
  });
});
