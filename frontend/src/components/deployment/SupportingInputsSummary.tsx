import { GitBranch, Waypoints } from 'lucide-react';

interface SupportingInputsSummaryProps {
  supportingModelIds: string[];
  strategyIds: string[];
}

const InputList = ({ label, values }: { label: string; values: string[] }): JSX.Element => (
  <div>
    <dt className="text-sm font-medium text-ink">{label}</dt>
    <dd className="mt-2 space-y-1 text-sm text-stone-600">
      {values.length === 0 ? (
        <span>None</span>
      ) : (
        values.map((value) => (
          <span key={value} className="block rounded-md border border-stone-200 bg-stone-50 px-2 py-1">
            {value}
          </span>
        ))
      )}
    </dd>
  </div>
);

/** Summary of supporting model and strategy inputs used by a core RL candidate. */
export default function SupportingInputsSummary({
  supportingModelIds,
  strategyIds,
}: SupportingInputsSummaryProps): JSX.Element {
  return (
    <section className="surface-panel p-5">
      <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-ink">
        <Waypoints size={16} aria-hidden="true" />
        Inputs
      </div>
      <dl className="grid gap-4 sm:grid-cols-2">
        <div className="flex gap-2">
          <GitBranch className="mt-0.5 shrink-0 text-stone-500" size={16} aria-hidden="true" />
          <InputList label="Supporting models" values={supportingModelIds} />
        </div>
        <div className="flex gap-2">
          <GitBranch className="mt-0.5 shrink-0 text-stone-500" size={16} aria-hidden="true" />
          <InputList label="Strategies" values={strategyIds} />
        </div>
      </dl>
    </section>
  );
}