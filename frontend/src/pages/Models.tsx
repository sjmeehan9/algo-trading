import { BrainCircuit, Plus } from 'lucide-react';
import { Link } from 'react-router-dom';

/** Route page for model registry browsing and creation entry points. */
export default function Models(): JSX.Element {
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">Models</h2>
          <p className="mt-1 text-sm text-stone-600">Core RL and supporting model configurations.</p>
        </div>
        <Link className="primary-button" to="/models/new">
          <Plus size={16} aria-hidden="true" />
          New model
        </Link>
      </div>

      <section className="surface-panel overflow-hidden">
        <div className="border-b border-stone-200 px-5 py-4">
          <h3 className="font-semibold text-ink">Registry</h3>
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
              <tr>
                <td colSpan={5} className="px-5 py-12 text-center text-stone-500">
                  <BrainCircuit className="mx-auto mb-3 text-stone-400" size={28} aria-hidden="true" />
                  No model configurations found.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}