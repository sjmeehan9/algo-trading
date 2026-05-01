import { configureStore } from '@reduxjs/toolkit';

import backtestReducer from './backtestSlice';
import modelsReducer from './modelsSlice';
import trainingReducer from './trainingSlice';

/** Global Redux store shared by the model-building frontend. */
export const store = configureStore({
  reducer: {
    models: modelsReducer,
    training: trainingReducer,
    backtest: backtestReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;