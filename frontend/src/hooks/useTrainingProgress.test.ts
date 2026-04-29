import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { wsClient } from '../api/websocket';
import { trainingProgressToHistoryPoint, useTrainingProgress } from './useTrainingProgress';

vi.mock('../api/websocket', () => ({
  wsClient: {
    subscribe: vi.fn(),
  },
}));

interface CapturedSubscription {
  topic: string;
  handler: (data: unknown) => void;
  unsubscribe: ReturnType<typeof vi.fn>;
}

const progressPayload = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({
  job_id: 'job-1',
  model_id: 'model-1',
  status: 'running',
  progress_percent: 25,
  current_timestep: 250,
  total_timesteps: 1000,
  current_metrics: { episode_reward: 4, loss: 0.25, learning_rate: 0.0003 },
  timestamp: '2026-04-29T00:00:00Z',
  ...overrides,
});

describe('useTrainingProgress', () => {
  let subscriptions: CapturedSubscription[];

  beforeEach(() => {
    subscriptions = [];
    vi.mocked(wsClient.subscribe).mockImplementation((topic, handler) => {
      const unsubscribe = vi.fn();
      subscriptions.push({ topic, handler: handler as (data: unknown) => void, unsubscribe });
      return unsubscribe;
    });
  });

  it('subscribes to job progress and stores chart history', () => {
    const { result, unmount } = renderHook(() => useTrainingProgress('model-1', 'job-1'));

    expect(subscriptions[0]?.topic).toBe('training:job-1');

    act(() => {
      subscriptions[0]?.handler(progressPayload());
    });

    expect(result.current.progress?.progress_percent).toBe(25);
    expect(result.current.metricsHistory).toEqual([
      { timestep: 250, reward: 4, loss: 0.25, learningRate: 0.0003 },
    ]);
    expect(result.current.isComplete).toBe(false);

    act(() => {
      subscriptions[0]?.handler(progressPayload({ status: 'completed', progress_percent: 100 }));
    });
    expect(result.current.isComplete).toBe(true);

    unmount();
    expect(subscriptions[0]?.unsubscribe).toHaveBeenCalledTimes(1);
  });

  it('keeps metrics history bounded', () => {
    const { result } = renderHook(() => useTrainingProgress('model-1', 'job-1'));

    act(() => {
      for (let index = 0; index < 505; index += 1) {
        subscriptions[0]?.handler(
          progressPayload({
            current_timestep: index,
            current_metrics: { episode_reward: index },
          }),
        );
      }
    });

    expect(result.current.metricsHistory).toHaveLength(500);
    expect(result.current.metricsHistory[0]?.timestep).toBe(5);
  });
});

describe('trainingProgressToHistoryPoint', () => {
  it('extracts alternate reward and loss metric names', () => {
    expect(
      trainingProgressToHistoryPoint({
        job_id: 'job-1',
        model_id: 'model-1',
        status: 'running',
        progress_percent: 50,
        current_timestep: 500,
        total_timesteps: 1000,
        current_metrics: { mean_reward: 12.25, policy_loss: 0.12 },
      }),
    ).toEqual({ timestep: 500, reward: 12.25, loss: 0.12 });
  });
});
