import { AlertTriangle, CheckCircle2, Loader2, XCircle } from 'lucide-react';

import type { DeploymentReadiness, ReadinessCheckStatus } from '../../api/deployment';

interface ReadinessChecklistProps {
  readiness: DeploymentReadiness | null;
  isLoading?: boolean;
}

const statusClass = (status: ReadinessCheckStatus): string => {
  if (status === 'passed') {
    return 'text-success';
  }
  if (status === 'warning') {
    return 'text-amber-700';
  }
  return 'text-red-700';
};

const statusIcon = (status: ReadinessCheckStatus): JSX.Element => {
  if (status === 'passed') {
    return <CheckCircle2 size={17} aria-hidden="true" />;
  }
  if (status === 'warning') {
    return <AlertTriangle size={17} aria-hidden="true" />;
  }
  return <XCircle size={17} aria-hidden="true" />;
};

/** Validation checklist for the selected deployment candidate. */
export default function ReadinessChecklist({
  readiness,
  isLoading = false,
}: ReadinessChecklistProps): JSX.Element {
  if (isLoading) {
    return (
      <section className="surface-panel flex items-center gap-2 p-5 text-sm text-stone-600">
        <Loader2 className="animate-spin" size={17} aria-hidden="true" />
        Validating readiness
      </section>
    );
  }

  if (!readiness) {
    return (
      <section className="surface-panel p-5 text-sm text-stone-500">
        Select a model generation to validate readiness.
      </section>
    );
  }

  return (
    <section className="surface-panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 px-5 py-4">
        <h3 className="font-semibold text-ink">Readiness checklist</h3>
        <span
          className={`status-pill ${
            readiness.deployable ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}
        >
          {readiness.deployable ? 'Deployable' : 'Blocked'}
        </span>
      </div>
      <div className="divide-y divide-stone-200">
        {readiness.checks.map((check) => (
          <div key={check.key} className="flex gap-3 px-5 py-3 text-sm">
            <span className={`mt-0.5 shrink-0 ${statusClass(check.status)}`}>
              {statusIcon(check.status)}
            </span>
            <div>
              <div className="font-medium text-ink">{check.label}</div>
              <div className="mt-1 text-stone-600">{check.message}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}