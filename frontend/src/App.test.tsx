import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { backtestingApi } from './api/backtesting';
import { modelsApi } from './api/models';
import { trainingApi } from './api/training';
import App from './App';

const createEmptyPaginated = () => ({
  items: [],
  total: 0,
  page: 1,
  page_size: 100,
  pages: 0,
});

vi.mock('./api/models', () => ({
  modelsApi: {
    list: vi.fn().mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 100,
      pages: 0,
    }),
    create: vi.fn(),
    get: vi.fn(),
    update: vi.fn(),
    remove: vi.fn(),
  },
  strategiesApi: {
    list: vi.fn().mockResolvedValue([]),
    get: vi.fn(),
  },
}));

vi.mock('./api/training', () => ({
  trainingApi: {
    listJobs: vi.fn().mockResolvedValue([]),
    createJob: vi.fn(),
    cancelJob: vi.fn(),
    reorderQueue: vi.fn(),
    listGenerations: vi.fn().mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 100,
      pages: 0,
    }),
  },
}));

vi.mock('./api/backtesting', () => ({
  backtestingApi: {
    run: vi.fn(),
    list: vi.fn().mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 100,
      pages: 0,
    }),
    get: vi.fn(),
    getTrades: vi.fn().mockResolvedValue([]),
    compare: vi.fn(),
  },
}));

vi.mock('./api/websocket', () => ({
  wsClient: {
    subscribe: vi.fn(() => vi.fn()),
  },
}));

describe('App', () => {
  beforeEach(() => {
    window.history.pushState({}, '', '/');
    vi.mocked(modelsApi.list).mockResolvedValue(createEmptyPaginated());
    vi.mocked(trainingApi.listJobs).mockResolvedValue([]);
    vi.mocked(trainingApi.listGenerations).mockResolvedValue(createEmptyPaginated());
    vi.mocked(backtestingApi.list).mockResolvedValue(createEmptyPaginated());
    vi.mocked(backtestingApi.getTrades).mockResolvedValue([]);
  });

  it('renders the dashboard route without crashing', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /models/i })).toBeInTheDocument();
  });

  it('navigates between planned routes', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('link', { name: /models/i }));
    expect(await screen.findByRole('heading', { level: 2, name: 'Models' })).toBeInTheDocument();

    await user.click(screen.getByRole('link', { name: /training/i }));
    expect(await screen.findByRole('heading', { level: 2, name: /Training/ })).toBeInTheDocument();

    await user.click(screen.getByRole('link', { name: /backtesting/i }));
    expect(
      await screen.findByRole('heading', { level: 2, name: 'Backtesting' }),
    ).toBeInTheDocument();
  });
});
