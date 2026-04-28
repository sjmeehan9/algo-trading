import { BarChart3, Calculator, Newspaper } from 'lucide-react';

import {
  FORBIDDEN_SUPPORTING_INPUT_TYPES,
  INPUT_DATA_TYPES,
  type InputDataTypeDef,
} from '../../constants/dataTypes';

interface DataTypeSelectorProps {
  selected: string[];
  onChange: (selected: string[]) => void;
  excludeTypes?: string[];
}

const iconByType: Record<InputDataTypeDef['icon'], typeof BarChart3> = {
  market: BarChart3,
  news: Newspaper,
  indicator: Calculator,
};

/** Multi-select control for supporting-model input data categories. */
export default function DataTypeSelector({
  selected,
  onChange,
  excludeTypes = FORBIDDEN_SUPPORTING_INPUT_TYPES,
}: DataTypeSelectorProps): JSX.Element {
  const excluded = new Set(excludeTypes.map((type) => type.trim().toLowerCase()));
  const visibleTypes = INPUT_DATA_TYPES.filter(
    (dataType) => !excluded.has(dataType.id.trim().toLowerCase()),
  );

  const handleToggle = (dataTypeId: string): void => {
    onChange(
      selected.includes(dataTypeId)
        ? selected.filter((selectedType) => selectedType !== dataTypeId)
        : [...selected, dataTypeId],
    );
  };

  return (
    <fieldset className="space-y-3">
      <legend className="sr-only">Input data types</legend>
      <div className="grid gap-3 md:grid-cols-3">
        {visibleTypes.map((dataType) => {
          const isSelected = selected.includes(dataType.id);
          const Icon = iconByType[dataType.icon];

          return (
            <label
              key={dataType.id}
              className={`flex min-h-[8rem] cursor-pointer flex-col gap-3 rounded-md border px-4 py-3 text-sm transition ${
                isSelected
                  ? 'border-action bg-blue-50 text-ink'
                  : 'border-stone-200 bg-white text-stone-700 hover:border-stone-300'
              }`}
            >
              <span className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => handleToggle(dataType.id)}
                  className="mt-1 h-4 w-4 rounded border-stone-300 text-action"
                />
                <span className="flex min-w-0 flex-1 flex-col gap-1">
                  <span className="flex items-center gap-2 font-medium text-ink">
                    <Icon size={16} aria-hidden="true" />
                    {dataType.label}
                  </span>
                  <span className="text-xs leading-5 text-stone-500">{dataType.description}</span>
                </span>
              </span>
              <span className="mt-auto flex flex-wrap gap-1">
                {dataType.frequencies.map((frequency) => (
                  <span
                    key={`${dataType.id}-${frequency}`}
                    className="rounded-md bg-stone-100 px-1.5 py-0.5 text-xs font-medium text-stone-600"
                  >
                    {frequency}
                  </span>
                ))}
              </span>
            </label>
          );
        })}
      </div>
      <p className="text-xs leading-5 text-stone-500">
        Supporting model signal inputs are intentionally unavailable here.
      </p>
    </fieldset>
  );
}
