import { Plus, X } from 'lucide-react';
import { useState } from 'react';

import { DATA_FREQUENCIES, DEFAULT_TRAINING_SYMBOLS } from '../../constants/algorithms';
import type { CoreRLModelFormData } from '../../hooks/useModelForm';

type TrainingDataFormValue = CoreRLModelFormData['training_data'];

interface DataSelectorProps {
  value: TrainingDataFormValue;
  onChange: (value: TrainingDataFormValue) => void;
}

const normalizeSymbol = (symbol: string): string => symbol.trim().toUpperCase();

/** Symbol, date range, and frequency selector for model training data. */
export default function DataSelector({ value, onChange }: DataSelectorProps): JSX.Element {
  const [customSymbol, setCustomSymbol] = useState('');

  const update = (changes: Partial<TrainingDataFormValue>): void => {
    onChange({ ...value, ...changes });
  };

  const toggleSymbol = (symbol: string): void => {
    const normalizedSymbol = normalizeSymbol(symbol);
    const nextSymbols = value.symbols.includes(normalizedSymbol)
      ? value.symbols.filter((selectedSymbol) => selectedSymbol !== normalizedSymbol)
      : [...value.symbols, normalizedSymbol];
    update({ symbols: nextSymbols });
  };

  const addCustomSymbol = (): void => {
    const normalizedSymbol = normalizeSymbol(customSymbol);
    if (!normalizedSymbol || value.symbols.includes(normalizedSymbol)) {
      setCustomSymbol('');
      return;
    }

    update({ symbols: [...value.symbols, normalizedSymbol] });
    setCustomSymbol('');
  };

  return (
    <div className="space-y-5">
      <fieldset>
        <legend className="mb-2 text-sm font-medium text-ink">Symbols</legend>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {DEFAULT_TRAINING_SYMBOLS.map((symbol) => (
            <label
              key={symbol}
              className={`flex items-center gap-3 rounded-md border px-3 py-2 text-sm transition ${
                value.symbols.includes(symbol)
                  ? 'border-action bg-blue-50 text-ink'
                  : 'border-stone-200 bg-white text-stone-700 hover:border-stone-300'
              }`}
            >
              <input
                type="checkbox"
                checked={value.symbols.includes(symbol)}
                onChange={() => toggleSymbol(symbol)}
                className="h-4 w-4 rounded border-stone-300 text-action"
              />
              {symbol}
            </label>
          ))}
        </div>
      </fieldset>

      <div className="flex flex-wrap gap-2">
        {value.symbols.map((symbol) => (
          <span
            key={symbol}
            className="inline-flex items-center gap-1 rounded-md bg-stone-100 px-2 py-1 text-sm text-ink"
          >
            {symbol}
            <button
              type="button"
              aria-label={`Remove ${symbol}`}
              onClick={() => toggleSymbol(symbol)}
              className="inline-flex h-5 w-5 items-center justify-center rounded-md text-stone-500 hover:bg-stone-200 hover:text-ink"
            >
              <X size={13} aria-hidden="true" />
            </button>
          </span>
        ))}
      </div>

      <div className="flex flex-col gap-2 sm:flex-row">
        <label htmlFor="custom-symbol" className="sr-only">
          Custom symbol
        </label>
        <input
          id="custom-symbol"
          value={customSymbol}
          onChange={(event) => setCustomSymbol(event.target.value.toUpperCase())}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              addCustomSymbol();
            }
          }}
          placeholder="Ticker"
          className="min-w-0 flex-1 rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
        />
        <button type="button" onClick={addCustomSymbol} className="secondary-button">
          <Plus size={16} aria-hidden="true" />
          Add
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div>
          <label htmlFor="training-start-date" className="mb-1 block text-sm font-medium text-ink">
            Start Date
          </label>
          <input
            id="training-start-date"
            type="date"
            value={value.start_date}
            onChange={(event) => update({ start_date: event.target.value })}
            className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
          />
        </div>

        <div>
          <label htmlFor="training-end-date" className="mb-1 block text-sm font-medium text-ink">
            End Date
          </label>
          <input
            id="training-end-date"
            type="date"
            value={value.end_date}
            onChange={(event) => update({ end_date: event.target.value })}
            className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
          />
        </div>

        <div>
          <label htmlFor="training-frequency" className="mb-1 block text-sm font-medium text-ink">
            Frequency
          </label>
          <select
            id="training-frequency"
            value={value.data_frequency}
            onChange={(event) => update({ data_frequency: event.target.value })}
            className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm text-ink shadow-sm focus:border-action"
          >
            {DATA_FREQUENCIES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
