import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';

import App from './App';

describe('App', () => {
  beforeEach(() => {
    window.history.pushState({}, '', '/');
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
    expect(await screen.findByRole('heading', { level: 2, name: 'Training' })).toBeInTheDocument();

    await user.click(screen.getByRole('link', { name: /backtesting/i }));
    expect(await screen.findByRole('heading', { level: 2, name: 'Backtesting' })).toBeInTheDocument();
  });
});