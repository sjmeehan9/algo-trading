import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiClientError } from '../../api/client';
import {
  modelsApi,
  type ModelConfigResponse,
  type SupportingModelLifecycleResponse,
} from '../../api/models';
import type * as ModelsModule from '../../api/models';
import SupportingModelLifecyclePanel from './SupportingModelLifecyclePanel';

vi.mock('../../api/models', async () => {
  const actual = await vi.importActual<typeof ModelsModule>('../../api/models');
  return {
    ...actual,
    modelsApi: {
      getLifecycle: vi.fn(),
      activatePretrained: vi.fn(),
      loadArtifact: vi.fn(),
      unload: vi.fn(),
    },
  };
});

const sentimentModel: ModelConfigResponse = {
  model_id: 'support-ml-1',
  name: 'News Sentiment',
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
  state: 'registered',
};

const registeredLifecycle: SupportingModelLifecycleResponse = {
  model_id: 'support-ml-1',
  model_type: 'supporting_ml',
  state: 'registered',
  model_path: null,
  trainer_class: 'NewsSentimentTrainer',
  input_data_types: ['news_text'],
  signal_type: 'sentiment',
  algorithm: 'transformer_sentiment',
  last_error: null,
  is_ready: false,
  readiness_checks: [
    { name: 'trainer_loaded', passed: false, detail: 'No trainer instance loaded' },
    { name: 'loaded_state', passed: false, detail: 'Current state is registered' },
  ],
};

const readyLifecycle: SupportingModelLifecycleResponse = {
  ...registeredLifecycle,
  state: 'ready',
  model_path: 'data/models/news_sentiment',
  is_ready: true,
  readiness_checks: [
    { name: 'trainer_loaded', passed: true, detail: 'Trainer instance attached' },
    { name: 'loaded_state', passed: true, detail: 'Current state is ready' },
  ],
};

const renderPanel = (model: ModelConfigResponse = sentimentModel): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <SupportingModelLifecyclePanel model={model} />
    </QueryClientProvider>,
  );
};

describe('SupportingModelLifecyclePanel', () => {
  beforeEach(() => {
    vi.mocked(modelsApi.getLifecycle).mockResolvedValue(registeredLifecycle);
    vi.mocked(modelsApi.activatePretrained).mockResolvedValue(readyLifecycle);
    vi.mocked(modelsApi.loadArtifact).mockResolvedValue(readyLifecycle);
    vi.mocked(modelsApi.unload).mockResolvedValue({
      ...registeredLifecycle,
      state: 'unloaded',
      is_ready: false,
    });
  });

  it('renders lifecycle state, signal type, input data types, and readiness checks', async () => {
    renderPanel();

    // Wait for the lifecycle query to resolve before asserting loaded fields.
    expect(await screen.findByText('sentiment')).toBeInTheDocument();
    expect(screen.getByText('registered')).toBeInTheDocument();
    expect(screen.getByText('news_text')).toBeInTheDocument();
    expect(screen.getByText('No artifact loaded')).toBeInTheDocument();
    expect(screen.getByText(/trainer_loaded/)).toBeInTheDocument();
  });

  it('activates a pretrained model and reflects the backend-returned ready state', async () => {
    const user = userEvent.setup();
    renderPanel();

    const activateButton = await screen.findByRole('button', { name: /Activate pretrained/ });
    expect(activateButton).toBeEnabled();

    // After activation the backend reports the new ready state on subsequent reads.
    vi.mocked(modelsApi.getLifecycle).mockResolvedValue(readyLifecycle);
    await user.click(activateButton);

    await waitFor(() =>
      expect(modelsApi.activatePretrained).toHaveBeenCalledWith('support-ml-1', {}),
    );
    // The shown state comes from the backend response, not an optimistic value.
    expect(await screen.findByText('data/models/news_sentiment')).toBeInTheDocument();
    expect(screen.getAllByText('ready')).not.toHaveLength(0);
  });

  it('surfaces a backend lifecycle error when activation is rejected', async () => {
    vi.mocked(modelsApi.activatePretrained).mockRejectedValue(
      new ApiClientError('Pretrained activation is not implemented for algorithm.', {
        statusCode: 409,
        errorCode: 'LIFECYCLE_OPERATION_ERROR',
      }),
    );
    const user = userEvent.setup();
    renderPanel();

    await user.click(await screen.findByRole('button', { name: /Activate pretrained/ }));

    expect(
      await screen.findByText('Pretrained activation is not implemented for algorithm.'),
    ).toBeInTheDocument();
    // State remains unchanged because the action failed.
    expect(screen.getByText('registered')).toBeInTheDocument();
  });

  it('requires an artifact path before submitting a load request', async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByLabelText(/Load external artifact/);
    await user.click(screen.getByRole('button', { name: /Load artifact/ }));

    expect(screen.getByText('Enter an artifact path before loading.')).toBeInTheDocument();
    expect(modelsApi.loadArtifact).not.toHaveBeenCalled();
  });

  it('loads an external artifact path and reflects ready state', async () => {
    const user = userEvent.setup();
    renderPanel();

    const pathInput = await screen.findByLabelText(/Load external artifact/);
    vi.mocked(modelsApi.getLifecycle).mockResolvedValue(readyLifecycle);
    await user.type(pathInput, 'data/models/news_sentiment');
    await user.click(screen.getByRole('button', { name: /Load artifact/ }));

    await waitFor(() =>
      expect(modelsApi.loadArtifact).toHaveBeenCalledWith('support-ml-1', {
        model_path: 'data/models/news_sentiment',
      }),
    );
    expect(await screen.findAllByText('ready')).not.toHaveLength(0);
  });

  it('disables activation for non-sentiment algorithms', async () => {
    vi.mocked(modelsApi.getLifecycle).mockResolvedValue({
      ...registeredLifecycle,
      algorithm: 'random_forest',
    });
    renderPanel({
      ...sentimentModel,
      algorithm: 'random_forest',
      trainer_type: 'sklearn',
    });

    // Wait for the lifecycle query to resolve before asserting on the algorithm.
    await screen.findByText('random_forest');
    expect(screen.getByRole('button', { name: /Activate pretrained/ })).toBeDisabled();
    expect(
      screen.getByText(/Pretrained activation supports sentiment backends only/),
    ).toBeInTheDocument();
  });

  it('allows unloading a ready model', async () => {
    vi.mocked(modelsApi.getLifecycle).mockResolvedValue(readyLifecycle);
    const user = userEvent.setup();
    renderPanel();

    const unloadButton = await screen.findByRole('button', { name: /Unload/ });
    await waitFor(() => expect(unloadButton).toBeEnabled());

    // After unload the backend reports the unloaded state on subsequent reads.
    vi.mocked(modelsApi.getLifecycle).mockResolvedValue({
      ...registeredLifecycle,
      state: 'unloaded',
      is_ready: false,
    });
    await user.click(unloadButton);

    await waitFor(() => expect(modelsApi.unload).toHaveBeenCalledWith('support-ml-1'));
    expect(await screen.findByText('unloaded')).toBeInTheDocument();
  });
});
