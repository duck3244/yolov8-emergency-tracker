import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, Download } from 'lucide-react';

interface SessionEntry {
  filename: string;
  kind: string;
  size_bytes: number;
  modified_at: string;
}

export function HistoryPanel() {
  const [items, setItems] = useState<SessionEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setBusy(true);
    setError(null);
    try {
      const r = await fetch('/api/sessions');
      if (!r.ok) throw new Error(`${r.status}`);
      const data = await r.json();
      setItems(data.items as SessionEntry[]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">History</CardTitle>
        <Button variant="ghost" size="icon" onClick={refresh} disabled={busy} aria-label="Refresh">
          <RefreshCw className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-2">
        {items.length === 0 && (
          <p className="text-xs text-muted-foreground">No saved sessions yet.</p>
        )}
        {items.slice(0, 50).map((it) => (
          <div
            key={it.filename}
            className="flex items-center justify-between gap-2 rounded border px-2 py-1.5 text-xs"
          >
            <div className="flex min-w-0 items-center gap-2">
              <Badge variant="outline" className="shrink-0">
                {it.kind}
              </Badge>
              <span className="truncate font-mono">{it.filename}</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="hidden sm:inline">{prettyBytes(it.size_bytes)}</span>
              <a
                href={`/api/sessions/${encodeURIComponent(it.filename)}`}
                className="inline-flex items-center gap-1 hover:text-foreground"
                title="Download"
              >
                <Download className="h-3.5 w-3.5" />
              </a>
            </div>
          </div>
        ))}
        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}

function prettyBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
