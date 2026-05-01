import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

import type { TrainingJob, TrainingJobStatus } from '../api/training';

export interface TrainingJobSummary {
  jobId: string;
  modelId: string;
  status: TrainingJobStatus;
  progressPercent: number;
}

interface TrainingState {
  activeJobId?: string;
  jobs: TrainingJobSummary[];
}

const initialState: TrainingState = {
  jobs: [],
};

const toTrainingJobSummary = (job: TrainingJob): TrainingJobSummary => ({
  jobId: job.job_id,
  modelId: job.model_id,
  status: job.status,
  progressPercent: job.progress_percent,
});

/** Redux slice that tracks active training job selection and summary state. */
export const trainingSlice = createSlice({
  name: 'training',
  initialState,
  reducers: {
    setActiveJob(state, action: PayloadAction<string | undefined>) {
      state.activeJobId = action.payload;
    },
    upsertJob(state, action: PayloadAction<TrainingJobSummary>) {
      const existingIndex = state.jobs.findIndex((job) => job.jobId === action.payload.jobId);

      if (existingIndex >= 0) {
        state.jobs[existingIndex] = action.payload;
        return;
      }

      state.jobs.unshift(action.payload);
    },
    upsertApiJob(state, action: PayloadAction<TrainingJob>) {
      const summary = toTrainingJobSummary(action.payload);
      const existingIndex = state.jobs.findIndex((job) => job.jobId === summary.jobId);

      if (existingIndex >= 0) {
        state.jobs[existingIndex] = summary;
        return;
      }

      state.jobs.unshift(summary);
    },
    clearTrainingState(state) {
      state.activeJobId = undefined;
      state.jobs = [];
    },
  },
});

export const { clearTrainingState, setActiveJob, upsertApiJob, upsertJob } = trainingSlice.actions;

export default trainingSlice.reducer;
