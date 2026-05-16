import { useEffect, useRef } from 'react';

/** 같은 값으로 빠르게 호출되어도 마지막 한 번만 ms 뒤에 실행한다. */
export function useDebouncedCallback<Args extends unknown[]>(
  fn: (...args: Args) => void,
  ms: number
) {
  const timer = useRef<number | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    []
  );

  return (...args: Args) => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => fnRef.current(...args), ms);
  };
}
