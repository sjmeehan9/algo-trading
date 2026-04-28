import type { HyperparameterDef, HyperparameterValue } from '../../constants/algorithms';

interface HyperparameterEditorProps {
  definitions: HyperparameterDef[];
  values: Record<string, HyperparameterValue>;
  onChange: (values: Record<string, HyperparameterValue>) => void;
}

const toNumericInputValue = (
  value: HyperparameterValue | undefined,
  fallback: HyperparameterValue,
): number => {
  const selectedValue = value ?? fallback;
  return typeof selectedValue === 'number' ? selectedValue : Number(selectedValue);
};

/** Dynamic editor for algorithm-specific hyperparameter fields. */
export default function HyperparameterEditor({
  definitions,
  values,
  onChange,
}: HyperparameterEditorProps): JSX.Element {
  const handleChange = (name: string, value: HyperparameterValue): void => {
    onChange({ ...values, [name]: value });
  };

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {definitions.map((definition) => {
        const inputId = `hyperparameter-${definition.name}`;
        const currentValue = values[definition.name] ?? definition.default;

        return (
          <div key={definition.name} className="space-y-1.5">
            <label htmlFor={inputId} className="block text-sm font-medium text-ink">
              {definition.label}
              <span className="ml-1 text-xs font-normal text-stone-500">{definition.name}</span>
            </label>

            {definition.type === 'number' && (
              <input
                id={inputId}
                type="number"
                value={toNumericInputValue(values[definition.name], definition.default)}
                min={definition.min}
                max={definition.max}
                step={definition.step}
                onChange={(event) => {
                  const nextValue = Number(event.target.value);
                  handleChange(
                    definition.name,
                    Number.isFinite(nextValue) ? nextValue : Number(definition.default),
                  );
                }}
                className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
              />
            )}

            {definition.type === 'select' && (
              <select
                id={inputId}
                value={String(currentValue)}
                onChange={(event) => handleChange(definition.name, event.target.value)}
                className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
              >
                {definition.options?.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            )}

            {definition.type === 'boolean' && (
              <label
                className="inline-flex items-center gap-2 text-sm text-stone-700"
                htmlFor={inputId}
              >
                <input
                  id={inputId}
                  type="checkbox"
                  checked={Boolean(currentValue)}
                  onChange={(event) => handleChange(definition.name, event.target.checked)}
                  className="h-4 w-4 rounded border-stone-300 text-action"
                />
                Enabled
              </label>
            )}

            <p className="text-xs leading-5 text-stone-500">{definition.description}</p>
          </div>
        );
      })}
    </div>
  );
}
