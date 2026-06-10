import { AlertCircle } from 'lucide-react';
import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';

import CoreRLModelForm from '../components/models/CoreRLModelForm';
import LoadingSpinner from '../components/common/LoadingSpinner';
import SupportingModelForm from '../components/models/SupportingModelForm';
import SupportingModelLifecyclePanel from '../components/models/SupportingModelLifecyclePanel';
import { modelsApi } from '../api/models';
import { type CoreRLModelFormData, useModelForm } from '../hooks/useModelForm';
import {
  type SupportingModelFormData,
  useSupportingModelForm,
} from '../hooks/useSupportingModelForm';

const errorToMessage = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unable to load model configuration.';

/** Route page for model configuration workflows. */
export default function ModelConfig(): JSX.Element {
  const { modelId } = useParams();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isSupportingCreateRoute =
    location.pathname.endsWith('/supporting/new') ||
    searchParams.get('model_type')?.startsWith('supporting') ||
    searchParams.get('kind') === 'supporting';
  const editableModelId = isSupportingCreateRoute ? undefined : modelId;

  const modelTypeQuery = useQuery({
    queryKey: ['models', editableModelId],
    queryFn: () => modelsApi.get(editableModelId ?? ''),
    enabled: Boolean(editableModelId),
  });

  const loadedModelType = modelTypeQuery.data?.model_type;
  const isSupportingForm =
    isSupportingCreateRoute ||
    loadedModelType === 'supporting_ml' ||
    loadedModelType === 'supporting_rl';
  const isCoreForm = !isSupportingForm;
  // The lifecycle panel only applies to a persisted supporting model being
  // edited (not the supporting create route, which has no model yet).
  const lifecycleModel =
    !isSupportingCreateRoute &&
    (loadedModelType === 'supporting_ml' || loadedModelType === 'supporting_rl')
      ? modelTypeQuery.data
      : undefined;

  const coreForm = useModelForm(editableModelId, { enabled: isCoreForm });
  const supportingForm = useSupportingModelForm(editableModelId, { enabled: isSupportingForm });

  const isResolvingModelType = Boolean(editableModelId) && modelTypeQuery.isLoading;
  const typeLoadError = modelTypeQuery.error ? errorToMessage(modelTypeQuery.error) : null;
  const activeLoadError =
    typeLoadError ?? (isSupportingForm ? supportingForm.loadError : coreForm.loadError);
  const isLoadingInitialData =
    isResolvingModelType ||
    (isSupportingForm ? supportingForm.isLoadingInitialData : coreForm.isLoadingInitialData);
  const modeLabel = useMemo(() => {
    if (isSupportingForm) {
      return supportingForm.isEditMode ? 'Edit supporting model' : 'New supporting model';
    }
    return coreForm.isEditMode ? 'Edit core RL model' : 'New core RL model';
  }, [coreForm.isEditMode, isSupportingForm, supportingForm.isEditMode]);
  const modeDescription = isSupportingForm
    ? 'Configure ML/RL signal models for core reinforcement-learning inputs.'
    : 'Configure algorithm, data, signals, and trading environment.';

  const handleSubmit = async (formData: CoreRLModelFormData): Promise<void> => {
    await coreForm.save(formData);
    navigate('/models');
  };

  const handleSupportingSubmit = async (formData: SupportingModelFormData): Promise<void> => {
    await supportingForm.save(formData);
    navigate('/models');
  };

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-ink">{modeLabel}</h2>
          <p className="mt-1 text-sm text-stone-600">{modeDescription}</p>
        </div>
      </div>

      {isLoadingInitialData && <LoadingSpinner />}

      {!isLoadingInitialData && activeLoadError && (
        <section
          className="surface-panel flex items-start gap-3 p-5 text-sm text-red-700"
          role="alert"
        >
          <AlertCircle className="mt-0.5 shrink-0" size={18} aria-hidden="true" />
          <div>
            <h3 className="font-semibold">Unable to load model configuration</h3>
            <p className="mt-1">{activeLoadError}</p>
          </div>
        </section>
      )}

      {!isLoadingInitialData && !activeLoadError && isSupportingForm && (
        <>
          {lifecycleModel && <SupportingModelLifecyclePanel model={lifecycleModel} />}
          <SupportingModelForm
            initialData={supportingForm.initialData}
            isLoading={supportingForm.isSaving}
            submitLabel={
              supportingForm.isEditMode ? 'Update Supporting Model' : 'Create Supporting Model'
            }
            errorMessage={supportingForm.saveError}
            onCancel={() => navigate('/models')}
            onSubmit={handleSupportingSubmit}
          />
        </>
      )}

      {!isLoadingInitialData && !activeLoadError && !isSupportingForm && (
        <CoreRLModelForm
          initialData={coreForm.initialData}
          isLoading={coreForm.isSaving}
          submitLabel={coreForm.isEditMode ? 'Update Model' : 'Create Model'}
          errorMessage={coreForm.saveError}
          onCancel={() => navigate('/models')}
          onSubmit={handleSubmit}
        />
      )}
    </div>
  );
}
