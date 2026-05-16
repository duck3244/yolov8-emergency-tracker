import { useEffect, useRef, useState } from 'react';
import { openStateSocket } from '@/api/client';
import type { State } from '@/api/types';

const DEFAULT_STATE: State = {
  running: false,
  source: null,
  session_id: null,
  frame_count: 0,
  fps: 0,
  counts: { entered: 0, exited: 0, current_inside: 0, area_name: 'Building', max_inside_seen: 0 },
  alert_status: 'normal',
  last_error: null,
};

export function useLiveState() {
  const [state, setState] = useState<State>(DEFAULT_STATE);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number>(0);

  useEffect(() => {
    // React StrictMode가 dev에서 effect를 즉시 마운트→언마운트→마운트한다.
    // 짧은 지연을 두면 첫 cleanup이 WS 핸드셰이크 전에 동작해
    // "closed before established" 콘솔 경고를 피할 수 있다.
    let intentionalClose = false;
    let openTimer: number | null = null;
    let retryTimer: number | null = null;

    function scheduleConnect(delayMs: number) {
      if (intentionalClose) return;
      if (openTimer !== null) window.clearTimeout(openTimer);
      openTimer = window.setTimeout(connect, delayMs);
    }

    function connect() {
      openTimer = null;
      if (intentionalClose) return;
      const ws = openStateSocket(
        (s) => {
          setState(s);
          setConnected(true);
          retryRef.current = 0;
        },
        () => {
          setConnected(false);
          if (intentionalClose) return;
          const delay = Math.min(5000, 500 * 2 ** retryRef.current);
          retryRef.current += 1;
          retryTimer = window.setTimeout(connect, delay);
        }
      );
      wsRef.current = ws;
    }

    scheduleConnect(50);

    return () => {
      intentionalClose = true;
      if (openTimer !== null) window.clearTimeout(openTimer);
      if (retryTimer !== null) window.clearTimeout(retryTimer);
      const ws = wsRef.current;
      // 핸드셰이크 중이면 onopen에서 즉시 close 하도록 위임
      if (ws && ws.readyState === WebSocket.CONNECTING) {
        ws.onopen = () => ws.close();
      } else {
        ws?.close();
      }
      wsRef.current = null;
    };
  }, []);

  return { state, connected };
}
