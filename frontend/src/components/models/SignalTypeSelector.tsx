import { Activity, Gauge, LineChart, MapPin, SlidersHorizontal, TrendingUp } from 'lucide-react';

import { SIGNAL_TYPES, type SignalTypeDef } from '../../constants/dataTypes';

interface SignalTypeSelectorProps {
  selected: string;
  onChange: (value: string) => void;
}

const iconsBySignalType: Record<string, typeof TrendingUp> = {
  sentiment: Activity,
  trend: TrendingUp,
  volatility: Gauge,
  position: MapPin,
  indicator: LineChart,
  custom: SlidersHorizontal,
};

const getSignalIcon = (signalType: SignalTypeDef): typeof TrendingUp =>
  iconsBySignalType[signalType.id] ?? SlidersHorizontal;

/** Radio-card selector for supporting model output signal categories. */
export default function SignalTypeSelector({
  selected,
  onChange,
}: SignalTypeSelectorProps): JSX.Element {
  return (
    <fieldset className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      <legend className="sr-only">Output signal type</legend>
      {SIGNAL_TYPES.map((signalType) => {
        const isSelected = selected === signalType.id;
        const Icon = getSignalIcon(signalType);
        const inputId = `signal-type-${signalType.id}`;

        return (
          <div
            key={signalType.id}
            className={`flex items-start gap-3 rounded-md border px-4 py-3 text-sm transition ${
              isSelected
                ? 'border-action bg-blue-50 text-ink'
                : 'border-stone-200 bg-white text-stone-700 hover:border-stone-300'
            }`}
          >
            <input
              id={inputId}
              type="radio"
              name="signal_type"
              value={signalType.id}
              checked={isSelected}
              onChange={() => onChange(signalType.id)}
              className="mt-1 h-4 w-4 border-stone-300 text-action"
            />
            <label htmlFor={inputId} className="min-w-0 flex-1 cursor-pointer">
              <span className="flex items-center gap-2 font-medium text-ink">
                <Icon size={16} aria-hidden="true" />
                {signalType.label}
              </span>
              <span className="mt-1 block text-xs leading-5 text-stone-500">
                {signalType.description}
              </span>
              <span className="mt-2 inline-flex rounded-md bg-stone-100 px-1.5 py-0.5 text-xs font-medium text-stone-600">
                {signalType.valueRange}
              </span>
            </label>
          </div>
        );
      })}
    </fieldset>
  );
}
