import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { deploymentApi, type DeploymentCandidate, type DeploymentSelection } from '../api/deployment';
import Deployment from './Deployment';

vi.mock('../api/deployment', () => ({
  deploymentApi: {
    listCandidates: vi.fn(),
    validate: vi.fn(),
    getSelection: vi.fn(),
    saveSelection: vi.fn(),
    clearSelection: vi.fn(),
  },
}));

const readiness = {
  deployable: true,
  errors: [],
  warnings: ['No optimizer history was found; manual review is recommended.'],
  checks: [
    {
      key: 'model_type',
      label: 'Core RL model',
      status: 'passed' as const,
      message: 'Model is a core RL deployment root.',
    },
    {
      key: 'evaluation_evidence',
      label: 'Evaluation evidence',
      status: 'passed' as const,
      message: 'Completed evaluation/backtest metrics are available.',
    },
  ],
};

const candidate: DeploymentCandidate = {
  model_id: 'core-1',
  name: 'Core PPO',
  description: 'Primary policy',
  algorithm: 'ppo',
  trainer_type: 'stable_baselines3',
  state: 'configured',
  supporting_model_ids: ['supporting-sentiment'],
  strategy_ids: ['strategy-position'],
  available_generations: [
    {
      generation_id: 'generation-2',
      generation_number: 2,
      status: 'evaluated',
      created_at: '2026-04-30T10:00:00Z',
      training_duration_seconds: 3600,
      final_reward: 2.4,
      model_path: 'models/core-1/generation-2.zip',
      metrics: {
        sharpe_ratio: 1.32,
        total_return: 0.14,
        max_drawdown: 0.05,
      },
      evaluation: {
        source: 'generation_evaluation',
        metrics: {
          sharpe_ratio: 1.32,
          total_return: 0.14,
          max_drawdown: 0.05,
        },
      },
    },
  ],
  latest_backtest: {
    source: 'generation_evaluation',
    metrics: {
      sharpe_ratio: 1.32,
      total_return: 0.14,
      max_drawdown: 0.05,
    },
  },
  readiness,
};

const savedSelection: DeploymentSelection = {
  model_id: 'core-1',
  generation_id: 'generation-2',
  selected_at: '2026-05-01T08:00:00Z',
  candidate,
  readiness,
  selected_by: 'frontend-test',
};

const renderDeployment = (): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Deployment />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

describe('Deployment', () => {
  beforeEach(() => {
    vi.mocked(deploymentApi.listCandidates).mockResolvedValue([candidate]);
    vi.mocked(deploymentApi.validate).mockResolvedValue(readiness);
    vi.mocked(deploymentApi.getSelection).mockResolvedValue(null);
    vi.mocked(deploymentApi.saveSelection).mockResolvedValue(savedSelection);
    vi.mocked(deploymentApi.clearSelection).mockResolvedValue({ status: 'cleared' });
  });

  it('renders the empty state when no core RL candidates exist', async () => {
    vi.mocked(deploymentApi.listCandidates).mockResolvedValue([]);

    renderDeployment();

    expect(await screen.findByText(/No core RL deployment candidates/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /New core model/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Training dashboard/ })).toBeInTheDocument();
  });

  it('selects a generation, validates readiness, and saves the candidate', async () => {
    const user = userEvent.setup();
    renderDeployment();

    expect(await screen.findByRole('button', { name: /Core PPO/ })).toBeInTheDocument();
    expect(await screen.findByLabelText('Generation')).toHaveValue('generation-2');
    expect(await screen.findByText('Readiness checklist')).toBeInTheDocument();
    expect(screen.getByText('Deployable')).toBeInTheDocument();

    const saveButton = screen.getByRole('button', { name: /Save candidate/ });
    await waitFor(() => expect(saveButton).not.toBeDisabled());
    await user.click(saveButton);

    await waitFor(() => expect(deploymentApi.saveSelection).toHaveBeenCalledTimes(1));
    expect(deploymentApi.saveSelection).toHaveBeenCalledWith({
      model_id: 'core-1',
      generation_id: 'generation-2',
    });
  });

  it('shows supporting models as inputs, not root candidates', async () => {
    renderDeployment();

    expect(await screen.findByRole('button', { name: /Core PPO/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Supporting Sentiment/ })).not.toBeInTheDocument();
    expect(screen.getByText('supporting-sentiment')).toBeInTheDocument();
    expect(screen.getByText('strategy-position')).toBeInTheDocument();
  });
});