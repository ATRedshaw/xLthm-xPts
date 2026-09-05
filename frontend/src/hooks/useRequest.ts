import { useCallback, useEffect, useState } from "react";

export function useRequest<T>(
  load: (signal: AbortSignal) => Promise<T>,
  dependencies: ReadonlyArray<unknown>,
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    load(controller.signal)
      .then((result) => {
        setData(result);
        setLoading(false);
      })
      .catch((requestError: Error) => {
        if (requestError.name !== "AbortError") {
          setError(requestError);
          setLoading(false);
        }
      });

    return () => controller.abort();
    // The caller controls refetches through its dependency list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, attempt]);

  return { data, error, loading, retry };
}
