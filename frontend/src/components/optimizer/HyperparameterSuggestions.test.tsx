import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { optimizerApi, type OptimizationResult } from '../../api/optimizer';
import HyperparameterSuggestions from './HyperparameterSuggestions';

vi.mock('../../api/optimizer', () => ({
  optimizerApi: {
    analyze: vi.fn(),
    getLatest: vi.fn(),
    apply: vi.fn(),
  },
}));

const baseSuggestion: OptimizationResult['suggestions'][number] = {
  suggestion_id: 'suggestion-1',
  parameter: 'learning_rate',
  current_value: 0.0003,
  suggested_value: 0.0001,
  rationale: 'Reward improved while update variance stayed elevated.',
  confidence: 'high',
  expected_impact: 'Smoother policy updates.',
  applied: false,
  applied_at: null,
  outcome_status: 'not_applied',
  outcome_notes: null,
};

const baseResult: OptimizationResult = {
  result_id: 'optimizer-1',
  model_id: 'core-1',
  created_at: '2026-04-29T12:00:00Z',
  analyzed_generation_ids: ['gen-1', 'gen-2'],
  source: 'openai',
  summary: 'Lower learning rate for the next PPO run.',
  priority_changes: ['learning_rate'],
  raw_response: null,
  suggestions: [baseSuggestion],
};

const appliedSuggestion: OptimizationResult['suggestions'][number] = {
  ...baseSuggestion,
  applied: true,
  applied_at: '2026-04-29T12:05:00Z',
  outcome_status: 'pending_next_generation',
  outcome_notes: 'Applied to model configuration; waiting for the next generation outcome.',
};

const appliedResult: OptimizationResult = {
  ...baseResult,
  suggestions: [appliedSuggestion],
};

const renderOptimizer = (modelId = 'core-1'): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <HyperparameterSuggestions modelId={modelId} />
    </QueryClientProvider>,
  );
};

describe('HyperparameterSuggestions', () => {
  beforeEach(() => {
    vi.mocked(optimizerApi.getLatest).mockResolvedValue(null);
    vi.mocked(optimizerApi.analyze).mockResolvedValue(baseResult);
    vi.mocked(optimizerApi.apply).mockResolvedValue({
      suggestion: appliedSuggestion,
      optimization_result: appliedResult,
      updated_model: {
        model_id: 'core-1',
        name: 'Core PPO',
        description: null,
        model_type: 'core_rl',
        signal_type: null,
        trainer_type: 'stable_baselines3',
        algorithm: 'ppo',
        hyperparameters: { learning_rate: 0.0001 },
        training_data_config: {},
        supporting_model_ids: [],
        strategy_ids: [],
        environment_config: {},
        reward_function: 'profit_seeker',
        input_data_types: [],
        input_frequency: null,
        created_at: '2026-04-01T00:00:00Z',
        updated_at: '2026-04-29T12:05:00Z',
        state: 'configured',
      },
    });
  });

  it('runs analysis and displays suggestions', async () => {
    const user = userEvent.setup();
    renderOptimizer();

    await user.click(await screen.findByRole('button', { name: /Analyze/ }));

    await waitFor(() => expect(optimizerApi.analyze).toHaveBeenCalledTimes(1));
    expect(optimizerApi.analyze).toHaveBeenCalledWith({
      model_id: 'core-1',
      max_generations: 5,
      include_backtest_metrics: true,
    });
    expect(await screen.findByText('Lower learning rate for the next PPO run.')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'learning_rate' })).toBeInTheDocument();
  });

  it('applies a suggestion and updates its state', async () => {
    const user = userEvent.setup();
    vi.mocked(optimizerApi.getLatest).mockResolvedValue(baseResult);
    renderOptimizer();

    await user.click(await screen.findByRole('button', { name: 'Apply' }));

    await waitFor(() => expect(optimizerApi.apply).toHaveBeenCalledTimes(1));
    expect(optimizerApi.apply).toHaveBeenCalledWith({
      model_id: 'core-1',
      result_id: 'optimizer-1',
      suggestion_id: 'suggestion-1',
    });
    expect(await screen.findByRole('button', { name: 'Applied' })).toBeDisabled();
  });
});