import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { HyperparameterDef } from '../../constants/algorithms';
import HyperparameterEditor from './HyperparameterEditor';

const definitions: HyperparameterDef[] = [
  {
    name: 'learning_rate',
    label: 'Learning Rate',
    type: 'number',
    default: 0.0003,
    min: 0.000001,
    max: 0.1,
    step: 0.0001,
    description: 'Step size for gradient updates.',
  },
  {
    name: 'policy_type',
    label: 'Policy Type',
    type: 'select',
    default: 'MlpPolicy',
    options: [
      { value: 'MlpPolicy', label: 'MLP' },
      { value: 'CnnPolicy', label: 'CNN' },
    ],
    description: 'Policy architecture.',
  },
  {
    name: 'normalize_advantage',
    label: 'Normalize Advantage',
    type: 'boolean',
    default: true,
    description: 'Normalize advantage estimates.',
  },
];

describe('HyperparameterEditor', () => {
  it('emits numeric value changes', () => {
    const handleChange = vi.fn();

    render(<HyperparameterEditor definitions={definitions} values={{}} onChange={handleChange} />);

    fireEvent.change(screen.getByLabelText(/Learning Rate/), { target: { value: '0.005' } });

    expect(handleChange).toHaveBeenCalledWith({ learning_rate: 0.005 });
  });

  it('emits select and boolean value changes while preserving existing values', () => {
    const handleChange = vi.fn();

    render(
      <HyperparameterEditor
        definitions={definitions}
        values={{ learning_rate: 0.001, policy_type: 'MlpPolicy', normalize_advantage: true }}
        onChange={handleChange}
      />,
    );

    fireEvent.change(screen.getByLabelText(/Policy Type/), { target: { value: 'CnnPolicy' } });
    fireEvent.click(screen.getByLabelText(/Normalize Advantage/));

    expect(handleChange).toHaveBeenNthCalledWith(1, {
      learning_rate: 0.001,
      policy_type: 'CnnPolicy',
      normalize_advantage: true,
    });
    expect(handleChange).toHaveBeenNthCalledWith(2, {
      learning_rate: 0.001,
      policy_type: 'MlpPolicy',
      normalize_advantage: false,
    });
  });
});
