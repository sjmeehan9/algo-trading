import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { PaginatedResponse } from '../api/client';
import {
  modelsApi,
  strategiesApi,
  type ModelConfigResponse,
  type StrategyInfo,
} from '../api/models';
import ModelConfig from './ModelConfig';

vi.mock('../api/models', () => ({
  modelsApi: {
    list: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    update: vi.fn(),
    remove: vi.fn(),
  },
  strategiesApi: {
    list: vi.fn(),
    get: vi.fn(),
  },
}));

const createPaginated = <Item,>(items: Item[]): PaginatedResponse<Item> => ({
  items,
  total: items.length,
  page: 1,
  page_size: 100,
  pages: items.length > 0 ? 1 : 0,
});

const supportingModel: ModelConfigResponse = {
  model_id: 'supporting-ml-1',
  name: 'FinBERT Sentiment',
  description: 'Financial news sentiment signal.',
  model_type: 'supporting_ml',
  signal_type: 'sentiment',
  trainer_type: 'huggingface',
  algorithm: 'transformer_sentiment',
  hyperparameters: {},
  training_data_config: {},
  supporting_model_ids: [],
  strategy_ids: [],
  environment_config: {},
  reward_function: null,
  input_data_types: ['news_text'],
  input_frequency: 'realtime',
  created_at: '2026-04-01T00:00:00Z',
  updated_at: '2026-04-02T00:00:00Z',
  state: 'ready',
};

const strategy: StrategyInfo = {
  strategy_id: 'strategy-1',
  name: 'Momentum Strategy',
  signal_type: 'trend',
  description: 'Momentum crossover signal.',
  version: '1.0.0',
  state: 'ready',
};

const createdModel: ModelConfigResponse = {
  ...supportingModel,
  model_id: 'core-rl-1',
  name: 'Core PPO',
  description: null,
  model_type: 'core_rl',
  signal_type: null,
  trainer_type: 'stable_baselines3',
  algorithm: 'ppo',
  hyperparameters: { learning_rate: 0.0003, total_timesteps: 100_000 },
  supporting_model_ids: ['supporting-ml-1'],
  strategy_ids: ['strategy-1'],
  reward_function: 'profit_seeker',
  input_data_types: [],
  input_frequency: null,
  state: 'configured',
};

const existingModel: ModelConfigResponse = {
  ...createdModel,
  name: 'Existing DQN',
  description: 'Existing model',
  algorithm: 'dqn',
  hyperparameters: {
    learning_rate: 0.0001,
    buffer_size: 1_000_000,
    batch_size: 32,
    gamma: 0.99,
    exploration_fraction: 0.1,
    target_update_interval: 10_000,
    total_timesteps: 200_000,
  },
  training_data_config: {
    symbols: ['MSFT'],
    start_date: '2025-01-01',
    end_date: '2025-03-31',
    data_frequency: '5m',
  },
  environment_config: {
    action_space: 'discrete',
    observation_window: 90,
    initial_cash: 150_000,
    transaction_cost_bps: 2,
    max_position_size: 0.75,
    allow_short_selling: false,
  },
};

const renderModelConfig = (route: string): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>
        <Routes>
          <Route path="/models/new" element={<ModelConfig />} />
          <Route path="/models/:modelId" element={<ModelConfig />} />
          <Route path="/models" element={<h2>Models</h2>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

describe('ModelConfig', () => {
  beforeEach(() => {
    vi.mocked(modelsApi.list).mockImplementation(async (params) => {
      if (params?.modelType === 'supporting_ml') {
        return createPaginated([supportingModel]);
      }
      return createPaginated<ModelConfigResponse>([]);
    });
    vi.mocked(strategiesApi.list).mockResolvedValue([strategy]);
    vi.mocked(modelsApi.create).mockResolvedValue(createdModel);
    vi.mocked(modelsApi.get).mockResolvedValue(existingModel);
    vi.mocked(modelsApi.update).mockResolvedValue({
      ...existingModel,
      name: 'Existing DQN Updated',
    });
  });

  it('creates a core RL model with supporting model and strategy inputs', async () => {
    const user = userEvent.setup();
    renderModelConfig('/models/new');

    await user.type(screen.getByLabelText(/Model Name/), 'Core PPO');
    await user.click(await screen.findByRole('checkbox', { name: /FinBERT Sentiment/ }));
    await user.click(await screen.findByRole('checkbox', { name: /Momentum Strategy/ }));
    await user.click(screen.getByRole('button', { name: /Create Model/ }));

    await waitFor(() => expect(modelsApi.create).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(modelsApi.create).mock.calls[0]?.[0];
    if (!payload) {
      throw new Error('Expected create payload.');
    }

    expect(payload).toMatchObject({
      name: 'Core PPO',
      model_type: 'core_rl',
      trainer_type: 'stable_baselines3',
      algorithm: 'ppo',
      supporting_model_ids: ['supporting-ml-1'],
      strategy_ids: ['strategy-1'],
      reward_function: 'profit_seeker',
    });
    expect(payload.hyperparameters).toMatchObject({ total_timesteps: 100_000 });
    expect(payload.training_data_config).toMatchObject({ symbols: ['SPY'] });
  });

  it('loads an existing model and submits updates', async () => {
    const user = userEvent.setup();
    renderModelConfig('/models/core-rl-1');

    const nameInput = await screen.findByDisplayValue('Existing DQN');
    expect(screen.getByLabelText(/Algorithm/)).toHaveValue('dqn');
    expect(await screen.findByRole('checkbox', { name: /FinBERT Sentiment/ })).toBeChecked();

    await user.clear(nameInput);
    await user.type(nameInput, 'Existing DQN Updated');
    await user.click(screen.getByRole('button', { name: /Update Model/ }));

    await waitFor(() => expect(modelsApi.update).toHaveBeenCalledTimes(1));
    const updateCall = vi.mocked(modelsApi.update).mock.calls[0];
    if (!updateCall) {
      throw new Error('Expected update call.');
    }

    expect(updateCall[0]).toBe('core-rl-1');
    expect(updateCall[1]).toMatchObject({
      name: 'Existing DQN Updated',
      algorithm: 'dqn',
      supporting_model_ids: ['supporting-ml-1'],
      strategy_ids: ['strategy-1'],
    });
  });
});
