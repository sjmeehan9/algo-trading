import { AlertCircle } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';

import CoreRLModelForm from '../components/models/CoreRLModelForm';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { type CoreRLModelFormData, useModelForm } from '../hooks/useModelForm';

/** Route page for model configuration workflows. */
export default function ModelConfig(): JSX.Element {
  const { modelId } = useParams();
  const navigate = useNavigate();
  const { initialData, isEditMode, isLoadingInitialData, loadError, saveError, isSaving, save } =
    useModelForm(modelId);

  const modeLabel = isEditMode ? 'Edit core RL model' : 'New core RL model';

  const handleSubmit = async (formData: CoreRLModelFormData): Promise<void> => {
    await save(formData);
    navigate('/models');
  };

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">{modeLabel}</h2>
          <p className="mt-1 text-sm text-stone-600">
            Configure algorithm, data, signals, and trading environment.
          </p>
        </div>
      </div>

      {isLoadingInitialData && <LoadingSpinner />}

      {!isLoadingInitialData && loadError && (
        <section
          className="surface-panel flex items-start gap-3 p-5 text-sm text-red-700"
          role="alert"
        >
          <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <div>
            <h3 className="font-semibold">Unable to load model configuration</h3>
            <p className="mt-1">{loadError}</p>
          </div>
        </section>
      )}

      {!isLoadingInitialData && !loadError && (
        <CoreRLModelForm
          initialData={initialData}
          isLoading={isSaving}
          submitLabel={isEditMode ? 'Update Model' : 'Create Model'}
          errorMessage={saveError}
          onCancel={() => navigate('/models')}
          onSubmit={handleSubmit}
        />
      )}
    </div>
  );
}
