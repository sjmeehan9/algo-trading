import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { PaginatedResponse } from '../api/client';
import { modelsApi, type ModelConfigResponse } from '../api/models';
import Models from './Models';

vi.mock('../api/models', () => ({
  modelsApi: {
    list: vi.fn(),
  },
}));

const coreModel: ModelConfigResponse = {
  model_id: 'core-rl-d4153568',
  name: 'PPO',
  description: 'test',
  model_type: 'core_rl',
  signal_type: null,
  trainer_type: 'stable_baselines3',
  algorithm: 'ppo',
  hyperparameters: { total_timesteps: 4096 },
  training_data_config: {
    symbols: ['SPY'],
    start_date: '2026-04-21',
    end_date: '2026-04-25',
    data_frequency: '1m',
  },
  supporting_model_ids: [],
  strategy_ids: [],
  environment_config: {},
  reward_function: 'profit_seeker',
  input_data_types: [],
  input_frequency: null,
  created_at: '2026-05-08T20:31:26.457817+00:00',
  updated_at: '2026-05-08T20:31:26.457817+00:00',
  state: 'configured',
};

const createPaginated = <Item,>(items: Item[]): PaginatedResponse<Item> => ({
  items,
  total: items.length,
  page: 1,
  page_size: 100,
  pages: items.length > 0 ? 1 : 0,
});

const renderModels = (): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Models />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

describe('Models', () => {
  beforeEach(() => {
    vi.mocked(modelsApi.list).mockResolvedValue(createPaginated([coreModel]));
  });

  it('renders persisted model registry rows from the API', async () => {
    renderModels();

    expect(await screen.findByRole('link', { name: 'PPO' })).toHaveAttribute(
      'href',
      '/models/core-rl-d4153568',
    );
    expect(screen.getByText('Core RL')).toBeInTheDocument();
    expect(screen.getByText('ppo')).toBeInTheDocument();
    expect(screen.getByText('configured')).toBeInTheDocument();
    expect(screen.getByText('1 model')).toBeInTheDocument();
    expect(modelsApi.list).toHaveBeenCalledWith({ pageSize: 100 });
  });
});