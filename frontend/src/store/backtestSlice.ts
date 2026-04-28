import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export type BacktestStatus = 'running' | 'completed' | 'failed';

export interface BacktestSummary {
  backtestId: string;
  modelId: string;
  generationId: string;
  status: BacktestStatus;
}

interface BacktestState {
  selectedBacktestIds: string[];
  results: BacktestSummary[];
}

const initialState: BacktestState = {
  selectedBacktestIds: [],
  results: [],
};

/** Redux slice that tracks selected and summarized backtest results. */
export const backtestSlice = createSlice({
  name: 'backtest',
  initialState,
  reducers: {
    setSelectedBacktests(state, action: PayloadAction<string[]>) {
      state.selectedBacktestIds = action.payload;
    },
    upsertBacktest(state, action: PayloadAction<BacktestSummary>) {
      const existingIndex = state.results.findIndex(
        (result) => result.backtestId === action.payload.backtestId,
      );

      if (existingIndex >= 0) {
        state.results[existingIndex] = action.payload;
        return;
      }

      state.results.unshift(action.payload);
    },
    clearBacktests(state) {
      state.selectedBacktestIds = [];
      state.results = [];
    },
  },
});

export const { clearBacktests, setSelectedBacktests, upsertBacktest } = backtestSlice.actions;

export default backtestSlice.reducer;