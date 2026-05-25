import { useQuery } from '@tanstack/react-query';
import { AlertCircle, BrainCircuit, Plus, Waypoints } from 'lucide-react';
import { Link } from 'react-router-dom';

import { modelsApi, type ModelConfigResponse, type ModelType } from '../api/models';

const modelTypeLabel = (modelType: ModelType): string => {
  switch (modelType) {
    case 'core_rl':
      return 'Core RL';
    case 'supporting_ml':
      return 'Supporting ML';
    case 'supporting_rl':
      return 'Supporting RL';
    default:
      return modelType;
  }
};

const formatUpdatedAt = (value: string): string => new Date(value).toLocaleString();

const modelCountLabel = (count: number): string => `${count} ${count === 1 ? 'model' : 'models'}`;

const renderRegistryRows = (models: ModelConfigResponse[]): JSX.Element => {
  if (models.length === 0) {
    return (
      <tr>
        <td colSpan={5} className="px-5 py-12 text-center text-stone-500">
          <BrainCircuit className="mx-auto mb-3 text-stone-400" size={28} aria-hidden="true" />
          No model configurations found.
        </td>
      </tr>
    );
  }

  return (
    <>
      {models.map((model) => (
        <tr key={model.model_id} className="transition hover:bg-stone-50">
          <td className="px-5 py-4 align-top">
            <Link to={`/models/${model.model_id}`} className="font-semibold text-action hover:underline">
              {model.name}
            </Link>
            {model.description && (
              <p className="mt-1 max-w-md truncate text-xs text-stone-500">{model.description}</p>
            )}
          </td>
          <td className="px-5 py-4 align-top text-stone-700">{modelTypeLabel(model.model_type)}</td>
          <td className="px-5 py-4 align-top text-stone-700">{model.algorithm}</td>
          <td className="px-5 py-4 align-top">
            <span className="status-pill bg-stone-100 text-stone-700">{model.state}</span>
          </td>
          <td className="px-5 py-4 align-top text-stone-600">{formatUpdatedAt(model.updated_at)}</td>
        </tr>
      ))}
    </>
  );
};

/** Route page for model registry browsing and creation entry points. */
export default function Models(): JSX.Element {
  const modelsQuery = useQuery({
    queryKey: ['models', 'registry'],
    queryFn: () => modelsApi.list({ pageSize: 100 }),
  });
  const models = modelsQuery.data?.items ?? [];

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">Models</h2>
          <p className="mt-1 text-sm text-stone-600">
            Core RL and supporting model configurations.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link className="secondary-button" to="/models/supporting/new">
            <Waypoints size={16} aria-hidden="true" />
            New supporting model
          </Link>
          <Link className="primary-button" to="/models/new">
            <Plus size={16} aria-hidden="true" />
            New core model
          </Link>
        </div>
      </div>

      {modelsQuery.error && (
        <section
          className="surface-panel flex items-start gap-3 p-4 text-sm text-red-700"
          role="alert"
        >
          <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <div>
            <h3 className="font-semibold">Unable to load model registry</h3>
            <p className="mt-1">Refresh the page after confirming the API is running.</p>
          </div>
        </section>
      )}

      <section className="surface-panel overflow-hidden">
        <div className="border-b border-stone-200 px-5 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="font-semibold text-ink">Registry</h3>
            <span className="status-pill bg-stone-100 text-stone-600">
              {modelCountLabel(modelsQuery.data?.total ?? 0)}
            </span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-stone-200 text-sm">
            <thead className="bg-stone-50 text-left text-xs uppercase text-stone-500">
              <tr>
                <th className="px-5 py-3 font-semibold">Name</th>
                <th className="px-5 py-3 font-semibold">Type</th>
                <th className="px-5 py-3 font-semibold">Algorithm</th>
                <th className="px-5 py-3 font-semibold">State</th>
                <th className="px-5 py-3 font-semibold">Updated</th>
              </tr>
            </thead>
            <tbody>
              {modelsQuery.isLoading ? (
                <tr>
                  <td colSpan={5} className="px-5 py-12 text-center text-stone-500">
                    Loading model registry...
                  </td>
                </tr>
              ) : (
                renderRegistryRows(models)
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
