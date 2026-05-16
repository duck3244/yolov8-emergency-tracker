import { useEffect, useState } from 'react';
import { api } from '@/api/client';
import { useLiveState } from '@/hooks/useLiveState';
import { useAlertSound } from '@/hooks/useAlertSound';
import { VideoPanel } from '@/components/VideoPanel';
import { CountsPanel } from '@/components/CountsPanel';
import { SourceSelect } from '@/components/SourceSelect';
import { SettingsForm } from '@/components/SettingsForm';
import { AreaEditor } from '@/components/AreaEditor';
import { AlertBanner } from '@/components/AlertBanner';
import { HistoryPanel } from '@/components/HistoryPanel';
import type { HealthResponse } from '@/api/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

type Tab = 'monitor' | 'areas' | 'settings' | 'history';

export default function App() {
  const { state, connected } = useLiveState();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [tab, setTab] = useState<Tab>('monitor');
  const [soundEnabled, setSoundEnabled] = useState<boolean>(() => {
    return localStorage.getItem('emt:sound') !== '0';
  });

  useAlertSound(soundEnabled, state.alert_status);

  function toggleSound(v: boolean) {
    setSoundEnabled(v);
    localStorage.setItem('emt:sound', v ? '1' : '0');
  }

  async function refreshHealth() {
    try {
      setHealth(await api.health());
    } catch {
      setHealth(null);
    }
  }

  useEffect(() => {
    refreshHealth();
    const t = setInterval(refreshHealth, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="container py-6 space-y-4">
      <header className="flex items-baseline justify-between border-b pb-3">
        <h1 className="text-2xl font-semibold">🚨 Emergency Tracker</h1>
        <div className="text-xs text-muted-foreground">
          {health
            ? `device: ${health.device ?? '—'} · uptime: ${health.uptime_seconds.toFixed(0)}s · model: ${health.model_ready ? 'ready' : 'loading'}`
            : 'backend offline'}
        </div>
      </header>

      <AlertBanner
        status={state.alert_status}
        inside={state.counts.current_inside}
        soundEnabled={soundEnabled}
        onSoundToggle={toggleSound}
      />

      <nav className="flex gap-1 border-b">
        {(['monitor', 'areas', 'settings', 'history'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm capitalize transition-colors ${
              tab === t
                ? 'border-b-2 border-primary font-medium text-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {t}
          </button>
        ))}
      </nav>

      {tab === 'monitor' && (
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-4">
            <VideoPanel />
            <CountsPanel state={state} connected={connected} />
          </div>
          <aside className="space-y-4">
            <SourceSelect running={state.running} onChange={refreshHealth} />
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Session</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>
                  Source: <span className="font-mono">{state.source ?? '—'}</span>
                </div>
                <div>
                  Session ID: <span className="font-mono">{state.session_id ?? '—'}</span>
                </div>
              </CardContent>
            </Card>
          </aside>
        </div>
      )}

      {tab === 'areas' && (
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <AreaEditor />
          </div>
          <aside>
            <CountsPanel state={state} connected={connected} />
          </aside>
        </div>
      )}

      {tab === 'settings' && (
        <div className="grid gap-4 lg:grid-cols-2">
          <SettingsForm />
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Location</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-1">
              <div>Name: <span className="font-mono">{health ? 'see config.json' : '—'}</span></div>
              <p className="text-xs text-muted-foreground">
                Location coordinates can be updated via <code>config.json</code> (not exposed in
                MVP UI).
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {tab === 'history' && (
        <div className="grid gap-4 lg:grid-cols-2">
          <HistoryPanel />
        </div>
      )}
    </div>
  );
}
