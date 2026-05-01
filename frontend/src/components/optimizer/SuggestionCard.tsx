import { CheckCircle2, Sparkles } from 'lucide-react';

import type { HyperparameterSuggestion } from '../../api/optimizer';

interface Props {
  suggestion: HyperparameterSuggestion;
  isApplying: boolean;
  onApply: () => void;
}

const confidenceClasses: Record<HyperparameterSuggestion['confidence'], string> = {
  high: 'bg-green-100 text-green-800',
  medium: 'bg-amber-100 text-amber-800',
  low: 'bg-stone-100 text-stone-700',
};

const formatValue = (value: HyperparameterSuggestion['suggested_value'] | null | undefined) => {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : value.toPrecision(4);
  }
  return String(value);
};

/** Display one hyperparameter suggestion and its application state. */
export default function SuggestionCard({ suggestion, isApplying, onApply }: Props): JSX.Element {
  return (
    <article className="rounded-md border border-stone-200 bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="font-semibold text-ink">{suggestion.parameter}</h4>
            <span className={`status-pill ${confidenceClasses[suggestion.confidence]}`}>
              {suggestion.confidence}
            </span>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-stone-600">
            <span>{formatValue(suggestion.current_value)}</span>
            <span aria-hidden="true">to</span>
            <span className="font-semibold text-ink">
              {formatValue(suggestion.suggested_value)}
            </span>
          </div>
        </div>
        <button
          type="button"
          className={suggestion.applied ? 'secondary-button' : 'primary-button'}
          disabled={suggestion.applied || isApplying}
          onClick={onApply}
        >
          {suggestion.applied ? (
            <CheckCircle2 size={16} aria-hidden="true" />
          ) : (
            <Sparkles size={16} aria-hidden="true" />
          )}
          {suggestion.applied ? 'Applied' : isApplying ? 'Applying' : 'Apply'}
        </button>
      </div>

      <p className="mt-3 text-sm text-stone-700">{suggestion.rationale}</p>
      <p className="mt-2 text-sm text-stone-600">{suggestion.expected_impact}</p>
      {suggestion.outcome_notes ? (
        <p className="mt-3 rounded-md bg-stone-50 px-3 py-2 text-xs text-stone-600">
          {suggestion.outcome_notes}
        </p>
      ) : null}
    </article>
  );
}
