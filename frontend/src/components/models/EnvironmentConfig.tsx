import { ACTION_SPACE_OPTIONS } from '../../constants/algorithms';
import type { CoreRLModelFormData } from '../../hooks/useModelForm';

type EnvironmentFormValue = CoreRLModelFormData['environment_config'];

interface EnvironmentConfigProps {
  value: EnvironmentFormValue;
  onChange: (value: EnvironmentFormValue) => void;
}

const numberValue = (value: string, fallback: number): number => {
  const parsedValue = Number(value);
  return Number.isFinite(parsedValue) ? parsedValue : fallback;
};

/** Trading environment controls for action, observation, capital, and cost settings. */
export default function EnvironmentConfig({
  value,
  onChange,
}: EnvironmentConfigProps): JSX.Element {
  const update = (changes: Partial<EnvironmentFormValue>): void => {
    onChange({ ...value, ...changes });
  };

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <div>
        <label
          htmlFor="environment-action-space"
          className="mb-1 block text-sm font-medium text-ink"
        >
          Action Space
        </label>
        <select
          id="environment-action-space"
          value={value.action_space}
          onChange={(event) => update({ action_space: event.target.value })}
          className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
        >
          {ACTION_SPACE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label
          htmlFor="environment-observation-window"
          className="mb-1 block text-sm font-medium text-ink"
        >
          Observation Window
        </label>
        <input
          id="environment-observation-window"
          type="number"
          min={1}
          max={10_000}
          step={1}
          value={value.observation_window}
          onChange={(event) =>
            update({
              observation_window: numberValue(event.target.value, value.observation_window),
            })
          }
          className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
        />
      </div>

      <div>
        <label
          htmlFor="environment-initial-cash"
          className="mb-1 block text-sm font-medium text-ink"
        >
          Initial Cash
        </label>
        <input
          id="environment-initial-cash"
          type="number"
          min={1_000}
          step={1_000}
          value={value.initial_cash}
          onChange={(event) =>
            update({ initial_cash: numberValue(event.target.value, value.initial_cash) })
          }
          className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
        />
      </div>

      <div>
        <label
          htmlFor="environment-transaction-cost"
          className="mb-1 block text-sm font-medium text-ink"
        >
          Transaction Cost (bps)
        </label>
        <input
          id="environment-transaction-cost"
          type="number"
          min={0}
          max={1_000}
          step={0.1}
          value={value.transaction_cost_bps}
          onChange={(event) =>
            update({
              transaction_cost_bps: numberValue(event.target.value, value.transaction_cost_bps),
            })
          }
          className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
        />
      </div>

      <div>
        <label
          htmlFor="environment-max-position"
          className="mb-1 block text-sm font-medium text-ink"
        >
          Max Position Size
        </label>
        <input
          id="environment-max-position"
          type="number"
          min={0.01}
          max={1}
          step={0.01}
          value={value.max_position_size}
          onChange={(event) =>
            update({ max_position_size: numberValue(event.target.value, value.max_position_size) })
          }
          className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
        />
      </div>

      <label className="flex items-center gap-3 self-end rounded-md border border-stone-200 px-3 py-2 text-sm text-ink">
        <input
          type="checkbox"
          checked={value.allow_short_selling}
          onChange={(event) => update({ allow_short_selling: event.target.checked })}
          className="h-4 w-4 rounded border-stone-300 text-action"
        />
        Allow short selling
      </label>
    </div>
  );
}
