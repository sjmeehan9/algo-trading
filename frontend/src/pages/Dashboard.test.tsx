import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { PaginatedResponse } from '../api/client';
import type { BacktestResult } from '../api/backtesting';
import { backtestingApi } from '../api/backtesting';
import { modelsApi, type ModelConfigResponse } from '../api/models';
import { trainingApi, type TrainingJob } from '../api/training';
import Dashboard from './Dashboard';

vi.mock('../api/models', () => ({
  modelsApi: {
    list: vi.fn(),
  },
}));

vi.mock('../api/training', () => ({
  trainingApi: {
    listJobs: vi.fn(),
  },
}));

vi.mock('../api/backtesting', () => ({
  backtestingApi: {
    list: vi.fn(),
  },
}));

const createPaginated = <Item,>(items: Item[], total = items.length): PaginatedResponse<Item> => ({
  items,
  total,
  page: 1,
  page_size: 1,
  pages: total > 0 ? total : 0,
});

const getSummaryValue = (label: string, value: string): HTMLElement => {
  const article = screen.getByText(label).closest('article');
  if (!article) {
    throw new Error(`Summary article for ${label} was not rendered.`);
  }
  return within(article).getByText(value);
};

const renderDashboard = (): void => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

describe('Dashboard', () => {
  beforeEach(() => {
    vi.mocked(modelsApi.list).mockResolvedValue(createPaginated<ModelConfigResponse>([], 3));
    vi.mocked(trainingApi.listJobs).mockResolvedValue([{}, {}] as TrainingJob[]);
    vi.mocked(backtestingApi.list).mockResolvedValue(createPaginated<BacktestResult>([], 4));
  });

  it('renders live summary counts from the API', async () => {
    renderDashboard();

    expect(await screen.findByText('3')).toBeInTheDocument();
    expect(getSummaryValue('Model configurations', '3')).toBeInTheDocument();
    expect(getSummaryValue('Training jobs', '2')).toBeInTheDocument();
    expect(getSummaryValue('Backtests', '4')).toBeInTheDocument();
    expect(modelsApi.list).toHaveBeenCalledWith({ pageSize: 1 });
    expect(backtestingApi.list).toHaveBeenCalledWith({ pageSize: 1 });
  });
});