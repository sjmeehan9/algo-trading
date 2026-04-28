import { createAsyncThunk, createSlice, type PayloadAction } from '@reduxjs/toolkit';

import { modelsApi, type ListModelsParams, type ModelConfigResponse } from '../api/models';

interface ModelsState {
  items: ModelConfigResponse[];
  total: number;
  page: number;
  pageSize: number;
  pages: number;
  status: 'idle' | 'loading' | 'succeeded' | 'failed';
  error?: string;
  selectedModelId?: string;
}

const initialState: ModelsState = {
  items: [],
  total: 0,
  page: 1,
  pageSize: 20,
  pages: 0,
  status: 'idle',
};

/** Load model configurations from the backend registry. */
export const fetchModels = createAsyncThunk('models/fetchModels', async (params?: ListModelsParams) =>
  modelsApi.list(params),
);

/** Redux slice that stores model registry list state and selection. */
export const modelsSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    selectModel(state, action: PayloadAction<string | undefined>) {
      state.selectedModelId = action.payload;
    },
    clearModelsError(state) {
      state.error = undefined;
      if (state.status === 'failed') {
        state.status = 'idle';
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchModels.pending, (state) => {
        state.status = 'loading';
        state.error = undefined;
      })
      .addCase(fetchModels.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.items = action.payload.items;
        state.total = action.payload.total;
        state.page = action.payload.page;
        state.pageSize = action.payload.page_size;
        state.pages = action.payload.pages;
      })
      .addCase(fetchModels.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message || 'Unable to load model configurations.';
      });
  },
});

export const { clearModelsError, selectModel } = modelsSlice.actions;

export default modelsSlice.reducer;