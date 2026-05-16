import { useEffect, useState } from 'react';
import { api } from '@/api/client';
import type { SourceItem } from '@/api/types';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Play, Square, RotateCcw, Save } from 'lucide-react';

export function SourceSelect({ running, onChange }: { running: boolean; onChange?: () => void }) {
  const [items, setItems] = useState<SourceItem[]>([]);
  const [selected, setSelected] = useState<string>('webcam');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.sources().then((r) => setItems(r.items)).catch((e) => setError(String(e)));
  }, []);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      await api.selectSource(selected);
      onChange?.();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function action(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChange?.();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Source & Controls</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Select value={selected} onValueChange={setSelected} disabled={busy}>
          <SelectTrigger>
            <SelectValue placeholder="Pick a source" />
          </SelectTrigger>
          <SelectContent>
            {items.map((it) => (
              <SelectItem key={it.id} value={it.id}>
                {it.label}
                {it.size_bytes ? ` (${(it.size_bytes / 1024 / 1024).toFixed(1)} MB)` : ''}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="flex flex-wrap gap-2">
          <Button onClick={start} disabled={busy}>
            <Play className="h-4 w-4" /> Start
          </Button>
          <Button variant="outline" onClick={() => action(api.stop)} disabled={busy || !running}>
            <Square className="h-4 w-4" /> Stop
          </Button>
          <Button variant="outline" onClick={() => action(api.resetCounts)} disabled={busy}>
            <RotateCcw className="h-4 w-4" /> Reset
          </Button>
          <Button variant="outline" onClick={() => action(api.saveState)} disabled={busy}>
            <Save className="h-4 w-4" /> Save state
          </Button>
        </div>

        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}
