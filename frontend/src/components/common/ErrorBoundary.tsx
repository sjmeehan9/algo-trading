import { Component, type ErrorInfo, type ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error?: Error;
}

/** React error boundary for isolating route-level rendering failures. */
export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public override state: ErrorBoundaryState = {};

  public static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  public override componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('Frontend render failure.', error, errorInfo);
  }

  public override render(): ReactNode {
    if (this.state.error) {
      return (
        <main className="flex min-h-screen items-center justify-center bg-panel p-6">
          <section className="surface-panel max-w-xl p-6">
            <p className="text-sm font-semibold text-red-700">Frontend render failure</p>
            <h1 className="mt-2 text-2xl font-semibold text-ink">The workspace could not load.</h1>
            <p className="mt-3 text-sm text-stone-600">{this.state.error.message}</p>
          </section>
        </main>
      );
    }

    return this.props.children;
  }
}