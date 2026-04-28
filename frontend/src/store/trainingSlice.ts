import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export type TrainingJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

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
    clearTrainingState(state) {
      state.activeJobId = undefined;
      state.jobs = [];
    },
  },
});

export const { clearTrainingState, setActiveJob, upsertJob } = trainingSlice.actions;

export default trainingSlice.reducer;