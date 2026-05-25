import { Activity, BarChart3, BrainCircuit, CheckCircle2, Play, Rocket } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';

import { backtestingApi } from '../api/backtesting';
import { modelsApi } from '../api/models';
import { trainingApi } from '../api/training';

const workflowItems = [
  { title: 'Configure', target: '/models/new', icon: BrainCircuit },
  { title: 'Train', target: '/training', icon: Play },
  { title: 'Evaluate', target: '/backtesting', icon: BarChart3 },
  { title: 'Deploy', target: '/deployment', icon: Rocket },
] as const;

/** Operational dashboard for the model-building frontend. */
export default function Dashboard(): JSX.Element {
  const modelsQuery = useQuery({
    queryKey: ['dashboard', 'models-summary'],
    queryFn: () => modelsApi.list({ pageSize: 1 }),
  });
  const trainingJobsQuery = useQuery({
    queryKey: ['dashboard', 'training-jobs-summary'],
    queryFn: () => trainingApi.listJobs(),
  });
  const backtestsQuery = useQuery({
    queryKey: ['dashboard', 'backtests-summary'],
    queryFn: () => backtestingApi.list({ pageSize: 1 }),
  });

  const summaryItems = [
    {
      label: 'Model configurations',
      value: modelsQuery.isLoading ? '-' : String(modelsQuery.data?.total ?? 0),
      accent: 'text-action',
      icon: BrainCircuit,
    },
    {
      label: 'Training jobs',
      value: trainingJobsQuery.isLoading ? '-' : String(trainingJobsQuery.data?.length ?? 0),
      accent: 'text-success',
      icon: Activity,
    },
    {
      label: 'Backtests',
      value: backtestsQuery.isLoading ? '-' : String(backtestsQuery.data?.total ?? 0),
      accent: 'text-caution',
      icon: BarChart3,
    },
  ] as const;

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6">
      <section className="grid gap-4 md:grid-cols-3">
        {summaryItems.map((item) => {
          const Icon = item.icon;

          return (
            <article key={item.label} className="surface-panel p-5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm font-medium text-stone-500">{item.label}</p>
                  <p className="mt-2 text-3xl font-semibold text-ink">{item.value}</p>
                </div>
                <Icon className={item.accent} size={26} aria-hidden="true" />
              </div>
            </article>
          );
        })}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.4fr_0.8fr]">
        <div className="surface-panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 pb-4">
            <div>
              <h2 className="text-lg font-semibold text-ink">Lifecycle</h2>
              <p className="mt-1 text-sm text-stone-600">Configuration, training, evaluation, deployment.</p>
            </div>
            <Link className="primary-button" to="/models/new">
              <BrainCircuit size={16} aria-hidden="true" />
              New model
            </Link>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {workflowItems.map((item) => {
              const Icon = item.icon;

              return (
                <Link
                  key={item.title}
                  to={item.target}
                  className="rounded-md border border-stone-200 bg-stone-50 p-4 transition hover:border-stone-300 hover:bg-white"
                >
                  <Icon className="text-stone-700" size={22} aria-hidden="true" />
                  <p className="mt-3 font-semibold text-ink">{item.title}</p>
                </Link>
              );
            })}
          </div>
        </div>

        <aside className="surface-panel p-5">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="text-success" size={24} aria-hidden="true" />
            <h2 className="text-lg font-semibold text-ink">Foundation Ready</h2>
          </div>
          <dl className="mt-5 space-y-4 text-sm">
            <div className="flex items-center justify-between gap-4">
              <dt className="text-stone-500">Routing</dt>
              <dd className="font-semibold text-success">Configured</dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-stone-500">State store</dt>
              <dd className="font-semibold text-success">Online</dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-stone-500">API client</dt>
              <dd className="font-semibold text-success">Authenticated</dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-stone-500">WebSocket</dt>
              <dd className="font-semibold text-success">Topic-ready</dd>
            </div>
          </dl>
        </aside>
      </section>
    </div>
  );
}