import type {
  Areas,
  ConfigPatch,
  ConfigSummary,
  HealthResponse,
  SourceItem,
  State,
} from './types';

const TOKEN: string | undefined = import.meta.env.VITE_APP_TOKEN;

function authHeaders(): Record<string, string> {
  return TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => jsonFetch<HealthResponse>('/healthz'),

  sources: () => jsonFetch<{ items: SourceItem[] }>('/api/sources'),
  selectSource: (source_id: string) =>
    jsonFetch<{ ok: boolean; message?: string }>('/api/sources/select', {
      method: 'POST',
      body: JSON.stringify({ source_id }),
    }),

  config: () => jsonFetch<ConfigSummary>('/api/config'),
  patchConfig: (patch: ConfigPatch) =>
    jsonFetch<ConfigSummary>('/api/config', {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),

  areas: () => jsonFetch<Areas>('/api/areas'),
  /** 부분 업데이트 — `entrance`나 `exit` 중 한 쪽만 보내도 됨. */
  putAreas: (payload: {
    entrance?: number[][];
    exit?: number[][];
    reset_counts?: boolean;
  }) =>
    jsonFetch<{ ok: boolean; message?: string }>('/api/areas', {
      method: 'PUT',
      body: JSON.stringify({ reset_counts: true, ...payload }),
    }),

  resetCounts: () =>
    jsonFetch<{ ok: boolean }>('/api/actions/reset_counts', { method: 'POST' }),
  stop: () => jsonFetch<{ ok: boolean }>('/api/actions/stop', { method: 'POST' }),
  saveState: () =>
    jsonFetch<{ ok: boolean }>('/api/actions/save_state', { method: 'POST' }),
  restoreState: () =>
    jsonFetch<{ ok: boolean }>('/api/actions/restore_state', { method: 'POST' }),
  sendTestAlert: () =>
    jsonFetch<{ ok: boolean }>('/api/actions/send_test_alert', { method: 'POST' }),
};

export function streamUrl(): string {
  // MJPEG는 헤더 사용이 까다로워 토큰을 쿼리 파라미터로 전달하지 않는다.
  // 대신 APP_TOKEN 비활성(로컬) 환경을 가정하거나 reverse proxy로 인증.
  return '/stream';
}

/** /api/snapshot URL. `key`를 명시적으로 바꿔야 캐시버스트 — 매 렌더마다 새 요청을
 *  보내지 않도록 Date.now() 사용을 피한다. */
export function snapshotUrl(key: number | string = 0): string {
  return `/api/snapshot?k=${encodeURIComponent(String(key))}`;
}

export function openStateSocket(onMessage: (s: State) => void, onClose?: () => void) {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const tokenQuery = TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : '';
  const url = `${proto}//${window.location.host}/ws/state${tokenQuery}`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data) as State);
    } catch {
      /* ignore */
    }
  };
  ws.onclose = () => onClose?.();
  return ws;
}
