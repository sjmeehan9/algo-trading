import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import {
  getDefaultSupportingModelFormValues,
  type SupportingModelFormData,
} from '../../hooks/useSupportingModelForm';
import SupportingModelForm from './SupportingModelForm';

const renderSupportingForm = (
  initialData: SupportingModelFormData = getDefaultSupportingModelFormValues('ml'),
) => {
  const handleSubmit = vi
    .fn<(data: SupportingModelFormData) => Promise<void>>()
    .mockResolvedValue(undefined);

  render(
    <SupportingModelForm initialData={initialData} onCancel={vi.fn()} onSubmit={handleSubmit} />,
  );

  return { handleSubmit };
};

describe('SupportingModelForm', () => {
  it('switches model category and resets framework, algorithm, and hyperparameters', async () => {
    const user = userEvent.setup();
    renderSupportingForm();

    expect(screen.getByLabelText(/Framework/)).toHaveValue('sklearn');
    expect(screen.getByLabelText(/Algorithm/)).toHaveValue('random_forest');
    expect(screen.getByLabelText(/Number of Trees/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /Reinforcement Learning/ }));

    expect(screen.getByLabelText(/Framework/)).toHaveValue('stable_baselines3');
    expect(screen.getByLabelText(/Algorithm/)).toHaveValue('ppo');
    expect(screen.getByLabelText(/Steps per Update/)).toBeInTheDocument();
  });

  it('filters algorithms by selected framework', async () => {
    const user = userEvent.setup();
    renderSupportingForm();

    await user.selectOptions(screen.getByLabelText(/Framework/), 'tensorflow');

    expect(screen.getByLabelText(/Algorithm/)).toHaveValue('lstm');
    expect(screen.getByLabelText(/LSTM Units/)).toBeInTheDocument();
  });

  it('submits selected data and signal types without exposing model signal inputs', async () => {
    const user = userEvent.setup();
    const { handleSubmit } = renderSupportingForm();

    expect(screen.queryByLabelText(/model signal/i)).not.toBeInTheDocument();

    await user.type(screen.getByLabelText(/Model Name/), 'News Sentiment Signal');
    await user.click(screen.getByRole('checkbox', { name: /News Text/ }));
    await user.click(screen.getByRole('radio', { name: /Trend/ }));
    await user.click(screen.getByRole('button', { name: /Save Supporting Model/ }));

    expect(handleSubmit).toHaveBeenCalledTimes(1);
    expect(handleSubmit.mock.calls[0]?.[0]).toMatchObject({
      name: 'News Sentiment Signal',
      input_data_types: ['news_text'],
      signal_type: 'trend',
    });
  });
});
