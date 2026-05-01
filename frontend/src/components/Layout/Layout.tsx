import { Suspense, useState } from 'react';
import { Outlet } from 'react-router-dom';

import LoadingSpinner from '../common/LoadingSpinner';
import Header from './Header';
import Sidebar from './Sidebar';

/** Main shell layout shared by all frontend routes. */
export default function Layout(): JSX.Element {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-panel text-ink">
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header onMenuClick={() => setIsSidebarOpen(true)} />
        <main className="flex-1 overflow-y-auto p-4 lg:p-6">
          <Suspense fallback={<LoadingSpinner />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  );
}