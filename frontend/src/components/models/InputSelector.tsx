import { useQuery } from '@tanstack/react-query';
import { CircleAlert } from 'lucide-react';

import {
  modelsApi,
  strategiesApi,
  type ModelConfigResponse,
  type StrategyInfo,
} from '../../api/models';
import LoadingSpinner from '../common/LoadingSpinner';

type InputSelectorType = 'models' | 'strategies';

interface InputSelectorProps {
  type: InputSelectorType;
  selected: string[];
  onChange: (selected: string[]) => void;
}

interface SelectableInput {
  id: string;
  name: string;
  description?: string | null;
  state: string;
  signalType?: string | null;
  badge: string;
}

const toModelInput = (model: ModelConfigResponse): SelectableInput => ({
  id: model.model_id,
  name: model.name,
  description: model.description,
  state: model.state,
  signalType: model.signal_type,
  badge: model.model_type === 'supporting_ml' ? 'ML' : 'RL',
});

const toStrategyInput = (strategy: StrategyInfo): SelectableInput => ({
  id: strategy.strategy_id,
  name: strategy.name,
  description: strategy.description,
  state: strategy.state,
  signalType: strategy.signal_type,
  badge: `Strategy ${strategy.version}`,
});

const supportingModelUnavailableReason = (state: string): string =>
  `Unavailable: train or load this supporting model before selecting it. Current state: ${state}.`;

const loadInputs = async (type: InputSelectorType): Promise<SelectableInput[]> => {
  if (type === 'strategies') {
    const strategies = await strategiesApi.list();
    return strategies.map(toStrategyInput);
  }

  const [supportingMlModels, supportingRlModels] = await Promise.all([
    modelsApi.list({ modelType: 'supporting_ml', pageSize: 100 }),
    modelsApi.list({ modelType: 'supporting_rl', pageSize: 100 }),
  ]);

  return [...supportingMlModels.items, ...supportingRlModels.items]
    .map(toModelInput)
    .sort((firstInput, secondInput) => firstInput.name.localeCompare(secondInput.name));
};

/** Multi-select control for supporting model and strategy signal inputs. */
export default function InputSelector({
  type,
  selected,
  onChange,
}: InputSelectorProps): JSX.Element {
  const {
    data: items = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ['core-rl-input-options', type],
    queryFn: () => loadInputs(type),
  });

  const handleToggle = (id: string): void => {
    onChange(
      selected.includes(id)
        ? selected.filter((selectedId) => selectedId !== id)
        : [...selected, id],
    );
  };

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
        <CircleAlert size={16} aria-hidden="true" />
        Unable to load {type === 'models' ? 'supporting models' : 'strategies'}.
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-stone-300 px-4 py-6 text-center text-sm text-stone-500">
        No {type === 'models' ? 'supporting models' : 'strategies'} available.
      </div>
    );
  }

  return (
    <div className="grid gap-3 md:grid-cols-2">
      {items.map((item) => {
        const isSelected = selected.includes(item.id);
        const isUnavailableModel = type === 'models' && item.state.toLowerCase() !== 'ready';
        const isDisabled = isUnavailableModel && !isSelected;
        const unavailableReason = isUnavailableModel
          ? supportingModelUnavailableReason(item.state)
          : null;
        const unavailableReasonId = `${type}-${item.id}-unavailable-reason`;

        return (
          <label
            key={item.id}
            className={`flex items-start gap-3 rounded-md border px-3 py-3 text-sm transition ${
              isSelected
                ? 'border-action bg-blue-50'
                : 'border-stone-200 bg-white hover:border-stone-300'
            } ${isDisabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
          >
            <input
              type="checkbox"
              checked={isSelected}
              disabled={isDisabled}
              aria-describedby={unavailableReason ? unavailableReasonId : undefined}
              onChange={() => handleToggle(item.id)}
              className="mt-1 h-4 w-4 rounded border-stone-300 text-action"
            />
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-center gap-2">
                <span className="font-medium text-ink">{item.name}</span>
                <span className="rounded-md bg-stone-100 px-1.5 py-0.5 text-xs font-semibold text-stone-600">
                  {item.badge}
                </span>
                {item.signalType && (
                  <span className="text-xs text-stone-500">{item.signalType}</span>
                )}
              </span>
              {item.description && (
                <span className="mt-1 block truncate text-xs text-stone-500">
                  {item.description}
                </span>
              )}
              {unavailableReason && (
                <span id={unavailableReasonId} className="mt-1 block text-xs text-caution">
                  {unavailableReason}
                </span>
              )}
            </span>
          </label>
        );
      })}
    </div>
  );
}
