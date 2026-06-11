import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { PaginatedResponse } from '../api/client';
import { modelsApi, type ModelConfigResponse } from '../api/models';
import { optimizerApi } from '../api/optimizer';
import { trainingApi, type GenerationSummary, type TrainingJob } from '../api/training';
import type * as TrainingModule from '../api/training';
import { wsClient } from '../api/websocket';
import Training from './Training';

vi.mock('../api/models', () => ({
  modelsApi: {
    list: vi.fn(),
  },
}));

vi.mock('../api/training', async () => {
  const actual = await vi.importActual<typeof TrainingModule>('../api/training');
  return {
    ...actual,
    trainingApi: {
      listJobs: vi.fn(),
      createJob: vi.fn(),
      cancelJob: vi.fn(),
      reorderQueue: vi.fn(),
      listGenerations: vi.fn(),
    },
  };
});

vi.mock('../api/optimizer', () => ({
  optimizerApi: {
    analyze: vi.fn(),
    getLatest: vi.fn(),
    apply: vi.fn(),
  },
}));

vi.mock('../api/websocket', () => ({
  wsClient: {
    subscribe: vi.fn(),
  },
}));

const coreModel: ModelConfigResponse = {
  model_id: 'core-1',
  name: 'Core PPO',
  description: 'Primary policy',
  model_type: 'core_rl',
  signal_type: null,
  trainer_type: 'stable_baselines3',
  algorithm: 'ppo',
  hyperparameters: { total_timesteps: 100_000 },
  training_data_config: {},
  supporting_model_ids: [],
  strategy_ids: [],
  environment_config: {},
  reward_function: 'profit_seeker',
  input_data_types: [],
  input_frequency: null,
  created_at: '2026-04-01T00:00:00Z',
  updated_at: '2026-04-02T00:00:00Z',
  state: 'configured',
};

const secondCoreModel: ModelConfigResponse = {
  ...coreModel,
  model_id: 'core-2',
  name: 'Core DQN',
  algorithm: 'dqn',
};

const supportingMlModel: ModelConfigResponse = {
  ...coreModel,
  model_id: 'support-ml-1',
  name: 'News Sentiment',
  model_type: 'supporting_ml',
  signal_type: 'sentiment',
  trainer_type: 'sklearn',
  algorithm: 'random_forest',
  reward_function: null,
  input_data_types: ['news_text'],
  input_frequency: 'irregular',
  state: 'registered',
};

const supportingRlModel: ModelConfigResponse = {
  ...coreModel,
  model_id: 'support-rl-1',
  name: 'Trend Signal',
  model_type: 'supporting_rl',
  signal_type: 'trend',
  trainer_type: 'stable_baselines3',
  algorithm: 'ppo',
  reward_function: null,
  input_data_types: ['market_bar'],
  input_frequency: '1m',
  state: 'ready',
};

const runningJob: TrainingJob = {
  job_id: 'job-running',
  model_id: 'core-1',
  status: 'running',
  created_at: '2026-04-29T09:00:00Z',
  started_at: '2026-04-29T09:01:00Z',
  completed_at: null,
  progress_percent: 10,
  current_timestep: 100,
  total_timesteps: 1000,
  current_metrics: { episode_reward: 1.5, loss: 0.9 },
  error_message: null,
  generation_id: 'gen-running',
  description: 'Smoke training run',
  training_config: {},
  data_config: {},
};

const firstQueuedJob: TrainingJob = {
  ...runningJob,
  job_id: 'job-queued-1',
  status: 'queued',
  started_at: null,
  progress_percent: 0,
  current_timestep: 0,
  generation_id: null,
};

const secondQueuedJob: TrainingJob = {
  ...firstQueuedJob,
  job_id: 'job-queued-2',
  model_id: 'core-2',
  total_timesteps: 2000,
  description: 'Second queue item',
};

const completedJob: TrainingJob = {
  ...runningJob,
  job_id: 'job-complete',
  status: 'completed',
  completed_at: '2026-04-29T09:20:00Z',
  progress_percent: 100,
};

const generations: GenerationSummary[] = [
  {
    generation_id: 'generation-2',
    generation_number: 2,
    model_id: 'core-1',
    created_at: '2026-04-29T08:00:00Z',
    training_duration_seconds: 3600,
    final_reward: 18.5,
    metrics: { sharpe_ratio: 1.4, total_return: 0.12 },
  },
  {
    generation_id: 'generation-1',
    generation_number: 1,
    model_id: 'core-1',
    created_at: '2026-04-28T08:00:00Z',
    training_duration_seconds: 1800,
    final_reward: 11.25,
    metrics: { sharpe_ratio: 0.7, total_return: 0.05 },
  },
];

const createPaginated = <Item,>(items: Item[]): PaginatedResponse<Item> => ({
  items,
  total: items.length,
  page: 1,
  page_size: 100,
  pages: items.length > 0 ? 1 : 0,
});

const renderTraining = (): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Training />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

describe('Training', () => {
  let websocketHandlers: Map<string, (data: unknown) => void>;

  beforeEach(() => {
    websocketHandlers = new Map();
    vi.mocked(modelsApi.list).mockResolvedValue(
      createPaginated([coreModel, secondCoreModel, supportingMlModel, supportingRlModel]),
    );
    vi.mocked(trainingApi.listJobs).mockImplementation(async (params) => {
      if (params?.status === 'running') {
        return [runningJob];
      }
      if (params?.status === 'queued') {
        return [firstQueuedJob, secondQueuedJob];
      }
      return [runningJob, firstQueuedJob, secondQueuedJob, completedJob];
    });
    vi.mocked(trainingApi.listGenerations).mockResolvedValue(createPaginated(generations));
    vi.mocked(trainingApi.createJob).mockResolvedValue(firstQueuedJob);
    vi.mocked(trainingApi.cancelJob).mockResolvedValue({
      ...firstQueuedJob,
      status: 'cancelled',
      completed_at: '2026-04-29T09:02:00Z',
    });
    vi.mocked(trainingApi.reorderQueue).mockResolvedValue([secondQueuedJob, firstQueuedJob]);
    vi.mocked(optimizerApi.getLatest).mockResolvedValue(null);
    vi.mocked(optimizerApi.analyze).mockRejectedValue(new Error('not used in training tests'));
    vi.mocked(wsClient.subscribe).mockImplementation((topic, handler) => {
      websocketHandlers.set(topic, handler as (data: unknown) => void);
      return vi.fn();
    });
  });

  it('renders live WebSocket progress for the active job', async () => {
    renderTraining();

    expect(await screen.findAllByText('Core PPO')).not.toHaveLength(0);
    expect(wsClient.subscribe).toHaveBeenCalledWith('training:job-running', expect.any(Function));

    act(() => {
      websocketHandlers.get('training:job-running')?.({
        ...runningJob,
        progress_percent: 50,
        current_timestep: 500,
        current_metrics: {
          episode_reward: 12.5,
          loss: 0.42,
          episode_length: 80,
          learning_rate: 0.0003,
        },
      });
    });

    expect(await screen.findByText(/Progress: 50.0%/)).toBeInTheDocument();
    expect(screen.getByText('12.50')).toBeInTheDocument();
    expect(screen.getByText('0.4200')).toBeInTheDocument();
  });

  it('includes supporting ML and RL models in the training selector with state and type', async () => {
    renderTraining();

    const selector = (await screen.findByLabelText(/Training model/)) as HTMLSelectElement;
    const optionText = Array.from(selector.options).map((option) => option.textContent ?? '');

    expect(optionText).toEqual(
      expect.arrayContaining([
        'Core PPO — Core RL · configured',
        'News Sentiment — Supporting ML · registered',
        'Trend Signal — Supporting RL · ready',
      ]),
    );

    // The supporting model entries expose both their type and lifecycle state.
    expect(
      optionText.some((text) => text.includes('Supporting ML') && text.includes('registered')),
    ).toBe(true);
    expect(
      optionText.some((text) => text.includes('Supporting RL') && text.includes('ready')),
    ).toBe(true);
  });

  it('starts a new training job from the dashboard', async () => {
    const user = userEvent.setup();
    renderTraining();

    await screen.findByLabelText(/Training model/);
    await user.clear(screen.getByLabelText(/Timesteps/));
    await user.type(screen.getByLabelText(/Timesteps/), '250000');
    await user.type(screen.getByLabelText(/Description/), 'Longer PPO run');
    const startButton = screen.getByRole('button', { name: /Start training/ });
    await waitFor(() => expect(startButton).not.toBeDisabled());
    await user.click(startButton);

    await waitFor(() => expect(trainingApi.createJob).toHaveBeenCalledTimes(1));
    expect(trainingApi.createJob).toHaveBeenCalledWith({
      model_id: 'core-1',
      total_timesteps: 250_000,
      description: 'Longer PPO run',
    });
  });

  it('continues training from a selected generation when opted in', async () => {
    const user = userEvent.setup();
    renderTraining();

    await screen.findByLabelText(/Training model/);
    // Wait for the model's generations to load so the toggle is enabled.
    const continueToggle = await screen.findByLabelText(/Continue training from a previous/);
    await waitFor(() => expect(continueToggle).not.toBeDisabled());
    await user.click(continueToggle);

    const generationSelect = await screen.findByLabelText(/Continue from generation/);
    await user.selectOptions(generationSelect, 'generation-1');

    const startButton = screen.getByRole('button', { name: /Start training/ });
    await waitFor(() => expect(startButton).not.toBeDisabled());
    await user.click(startButton);

    await waitFor(() => expect(trainingApi.createJob).toHaveBeenCalledTimes(1));
    expect(trainingApi.createJob).toHaveBeenCalledWith(
      expect.objectContaining({
        model_id: 'core-1',
        continue_from_generation_id: 'generation-1',
      }),
    );
  });

  it('does not send a continue id for a fresh run', async () => {
    const user = userEvent.setup();
    renderTraining();

    await screen.findByLabelText(/Training model/);
    const startButton = screen.getByRole('button', { name: /Start training/ });
    await waitFor(() => expect(startButton).not.toBeDisabled());
    await user.click(startButton);

    await waitFor(() => expect(trainingApi.createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(trainingApi.createJob).mock.calls[0]?.[0];
    expect(payload).not.toHaveProperty('continue_from_generation_id');
  });

  it('displays running and completed job state from API payloads', async () => {
    renderTraining();

    expect(await screen.findByText(/Progress: 10.0%/)).toBeInTheDocument();
    expect(screen.getByText('100 / 1,000 timesteps')).toBeInTheDocument();
    expect(screen.getByText('job-complete')).toBeInTheDocument();
    expect(screen.getByText('completed')).toBeInTheDocument();
  });

  it('reorders and cancels queued jobs', async () => {
    const user = userEvent.setup();
    renderTraining();

    await user.click(await screen.findByRole('button', { name: /Move Core PPO down/ }));
    await waitFor(() => expect(trainingApi.reorderQueue).toHaveBeenCalledTimes(1));
    expect(trainingApi.reorderQueue).toHaveBeenCalledWith(['job-queued-2', 'job-queued-1']);

    await user.click(screen.getByRole('button', { name: /Cancel queued job for Core DQN/ }));
    await waitFor(() => expect(trainingApi.cancelJob).toHaveBeenCalledWith('job-queued-2'));
  });

  it('loads generation history and compares selected generations', async () => {
    const user = userEvent.setup();
    renderTraining();

    await user.click(await screen.findByRole('checkbox', { name: /Compare Generation 2/ }));
    await user.click(screen.getByRole('checkbox', { name: /Compare Generation 1/ }));

    expect(screen.getAllByText('Gen 2')).not.toHaveLength(0);
    expect(screen.getByRole('heading', { name: /Generation comparison/ })).toBeInTheDocument();
  });
});
