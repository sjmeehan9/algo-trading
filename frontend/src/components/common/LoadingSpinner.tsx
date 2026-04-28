/** Accessible loading indicator used by route and data-loading boundaries. */
export default function LoadingSpinner(): JSX.Element {
  return (
    <div className="flex min-h-40 items-center justify-center" role="status" aria-live="polite">
      <div className="h-10 w-10 animate-spin rounded-full border-4 border-stone-200 border-t-action" />
      <span className="sr-only">Loading</span>
    </div>
  );
}