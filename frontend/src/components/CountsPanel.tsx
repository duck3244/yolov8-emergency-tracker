import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { State } from '@/api/types';

interface Props {
  state: State;
  connected: boolean;
}

const ALERT_LABEL: Record<State['alert_status'], { label: string; variant: 'success' | 'warning' | 'destructive' }> = {
  normal: { label: 'NORMAL', variant: 'success' },
  warning: { label: 'WARNING', variant: 'warning' },
  emergency: { label: 'EMERGENCY', variant: 'destructive' },
};

export function CountsPanel({ state, connected }: Props) {
  const alert = ALERT_LABEL[state.alert_status];
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base">Current Counts</CardTitle>
        <div className="flex items-center gap-2">
          <Badge variant={alert.variant}>{alert.label}</Badge>
          <Badge variant={connected ? 'secondary' : 'destructive'}>
            {connected ? 'WS connected' : 'WS disconnected'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4 text-center">
          <Stat label="Entered" value={state.counts.entered} />
          <Stat label="Exited" value={state.counts.exited} />
          <Stat label="Inside" value={state.counts.current_inside} highlight />
        </div>
        <div className="mt-4 grid grid-cols-3 gap-4 text-sm text-muted-foreground">
          <div>FPS: <span className="font-mono">{state.fps.toFixed(1)}</span></div>
          <div>Frames: <span className="font-mono">{state.frame_count}</span></div>
          <div>Max inside: <span className="font-mono">{state.counts.max_inside_seen}</span></div>
        </div>
        {state.last_error && (
          <div className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
            {state.last_error}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div>
      <div className={`text-3xl font-bold ${highlight ? 'text-primary' : ''}`}>{value}</div>
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
    </div>
  );
}
