import { lazy } from 'react';
import { createBrowserRouter } from 'react-router-dom';

import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard';

const ModelsPage = lazy(() => import('./pages/Models'));
const ModelConfigPage = lazy(() => import('./pages/ModelConfig'));
const TrainingPage = lazy(() => import('./pages/Training'));
const BacktestingPage = lazy(() => import('./pages/Backtesting'));
const DeploymentPage = lazy(() => import('./pages/Deployment'));

/** Browser router covering all planned model lifecycle pages. */
export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'models/new', element: <ModelConfigPage /> },
      { path: 'models/:modelId', element: <ModelConfigPage /> },
      { path: 'models/:modelId/supporting/new', element: <ModelConfigPage /> },
      { path: 'training', element: <TrainingPage /> },
      { path: 'backtesting', element: <BacktestingPage /> },
      { path: 'deployment', element: <DeploymentPage /> },
    ],
  },
]);