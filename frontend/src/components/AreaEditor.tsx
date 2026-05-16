import { useEffect, useRef, useState } from 'react';
import { api, snapshotUrl } from '@/api/client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Plus, RotateCcw, Save, MousePointer2 } from 'lucide-react';

type Mode = 'entrance' | 'exit';

const COLORS: Record<Mode, { stroke: string; fill: string; label: string }> = {
  entrance: { stroke: '#22c55e', fill: 'rgba(34, 197, 94, 0.25)', label: 'Entrance' },
  exit: { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.25)', label: 'Exit' },
};

export function AreaEditor() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [mode, setMode] = useState<Mode>('entrance');
  const [entrance, setEntrance] = useState<number[][]>([]);
  const [exitPoly, setExitPoly] = useState<number[][]>([]);
  const [confirmOpen, setConfirmOpen] = useState<null | 'both' | 'entrance' | 'exit'>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snapshotKey, setSnapshotKey] = useState(0);

  // 초기 데이터: 현재 영역 + 스냅샷
  useEffect(() => {
    api
      .areas()
      .then((a) => {
        setEntrance(a.entrance);
        setExitPoly(a.exit);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // snapshot URL은 snapshotKey가 바뀔 때만 변경 — 부모 재렌더링으로 인한
  // /api/snapshot 폭주를 방지.
  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    el.src = snapshotUrl(snapshotKey);
  }, [snapshotKey]);

  // 캔버스 그리기
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imgSize) return;
    canvas.width = imgSize.w;
    canvas.height = imgSize.h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    drawPolygon(ctx, entrance, COLORS.entrance, mode === 'entrance');
    drawPolygon(ctx, exitPoly, COLORS.exit, mode === 'exit');
  }, [entrance, exitPoly, mode, imgSize, snapshotKey]);

  function handleSnapshotLoaded() {
    const img = imgRef.current;
    if (!img) return;
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
  }

  function clientToImage(ev: React.MouseEvent<HTMLCanvasElement>): [number, number] | null {
    const canvas = canvasRef.current;
    if (!canvas || !imgSize) return null;
    const rect = canvas.getBoundingClientRect();
    // CSS 표시 크기 → 원본 픽셀로 스케일 역변환
    const scaleX = imgSize.w / rect.width;
    const scaleY = imgSize.h / rect.height;
    const x = Math.round((ev.clientX - rect.left) * scaleX);
    const y = Math.round((ev.clientY - rect.top) * scaleY);
    if (x < 0 || y < 0 || x > imgSize.w || y > imgSize.h) return null;
    return [x, y];
  }

  function handleClick(ev: React.MouseEvent<HTMLCanvasElement>) {
    const pt = clientToImage(ev);
    if (!pt) return;
    if (mode === 'entrance') setEntrance((prev) => [...prev, pt]);
    else setExitPoly((prev) => [...prev, pt]);
  }

  function clearCurrent() {
    if (mode === 'entrance') setEntrance([]);
    else setExitPoly([]);
  }

  function undoCurrent() {
    if (mode === 'entrance') setEntrance((p) => p.slice(0, -1));
    else setExitPoly((p) => p.slice(0, -1));
  }

  async function save(scope: 'both' | 'entrance' | 'exit', resetCounts: boolean) {
    setBusy(true);
    setError(null);
    try {
      const payload: Parameters<typeof api.putAreas>[0] = { reset_counts: resetCounts };
      if (scope === 'both' || scope === 'entrance') {
        if (entrance.length < 3) throw new Error('Entrance needs at least 3 points');
        payload.entrance = entrance;
      }
      if (scope === 'both' || scope === 'exit') {
        if (exitPoly.length < 3) throw new Error('Exit needs at least 3 points');
        payload.exit = exitPoly;
      }
      await api.putAreas(payload);
      setConfirmOpen(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const overlap = polygonsOverlap(entrance, exitPoly);
  const canSaveBoth = entrance.length >= 3 && exitPoly.length >= 3;
  const canSaveEntrance = entrance.length >= 3;
  const canSaveExit = exitPoly.length >= 3;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base flex items-center gap-2">
          <MousePointer2 className="h-4 w-4" /> Area Editor
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge
            variant={mode === 'entrance' ? 'success' : 'outline'}
            className="cursor-pointer"
            onClick={() => setMode('entrance')}
          >
            Entrance ({entrance.length})
          </Badge>
          <Badge
            variant={mode === 'exit' ? 'destructive' : 'outline'}
            className="cursor-pointer"
            onClick={() => setMode('exit')}
          >
            Exit ({exitPoly.length})
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="relative w-full bg-black rounded-md overflow-hidden">
          {/* src는 위의 useEffect에서 snapshotKey 변경 시에만 세팅 — 폭주 방지 */}
          <img
            ref={imgRef}
            alt="snapshot"
            onLoad={handleSnapshotLoaded}
            onError={() => {
              setImgSize(null);
              setError('Snapshot unavailable — start a source then click "Refresh snapshot"');
            }}
            className="hidden"
          />
          {imgSize ? (
            <canvas
              ref={canvasRef}
              onClick={handleClick}
              className="w-full h-auto cursor-crosshair"
            />
          ) : (
            <div className="aspect-video flex items-center justify-center text-muted-foreground text-sm">
              No frame yet — start a source then click "Refresh snapshot"
            </div>
          )}
        </div>

        {overlap && (
          <p className="text-xs text-amber-600">
            ⚠ Polygons overlap. State machine will keep previous area on overlap — allowed but may
            cause unstable transitions.
          </p>
        )}

        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={() => setSnapshotKey((k) => k + 1)}>
            <RotateCcw className="h-4 w-4" /> Refresh snapshot
          </Button>
          <Button variant="outline" size="sm" onClick={undoCurrent}>
            Undo last point
          </Button>
          <Button variant="outline" size="sm" onClick={clearCurrent}>
            Clear {COLORS[mode].label}
          </Button>
          <div className="flex-1" />
          <Button
            variant="outline"
            disabled={!canSaveEntrance || busy}
            size="sm"
            onClick={() => setConfirmOpen('entrance')}
          >
            <Save className="h-4 w-4" /> Save Entrance
          </Button>
          <Button
            variant="outline"
            disabled={!canSaveExit || busy}
            size="sm"
            onClick={() => setConfirmOpen('exit')}
          >
            <Save className="h-4 w-4" /> Save Exit
          </Button>
          <Button
            disabled={!canSaveBoth || busy}
            size="sm"
            onClick={() => setConfirmOpen('both')}
          >
            <Save className="h-4 w-4" /> Save Both
          </Button>
        </div>

        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>

      <Dialog open={confirmOpen !== null} onOpenChange={(o) => !o && setConfirmOpen(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {confirmOpen === 'both' && 'Save both polygons?'}
              {confirmOpen === 'entrance' && 'Save Entrance only?'}
              {confirmOpen === 'exit' && 'Save Exit only?'}
            </DialogTitle>
            <DialogDescription>
              Saving will <strong>reset entry/exit counts</strong> by default — areas changed
              mid-session would otherwise mix measurements from different geometries. Use "Keep counts"
              only when you're certain the new geometry preserves the same logical zone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => confirmOpen && save(confirmOpen, false)}
            >
              Keep counts
            </Button>
            <Button disabled={busy} onClick={() => confirmOpen && save(confirmOpen, true)}>
              <Plus className="h-4 w-4" /> Save & reset counts
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function drawPolygon(
  ctx: CanvasRenderingContext2D,
  points: number[][],
  color: { stroke: string; fill: string },
  active: boolean
) {
  if (points.length === 0) return;
  ctx.lineWidth = active ? 3 : 2;
  ctx.strokeStyle = color.stroke;
  ctx.fillStyle = color.fill;

  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  if (points.length >= 3) ctx.closePath();
  if (points.length >= 3) ctx.fill();
  ctx.stroke();

  // 점 표시
  for (const [x, y] of points) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffff00';
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

/** 단순 bbox 겹침 검사 (정확한 폴리곤 겹침이 아닌 빠른 휴리스틱). */
function polygonsOverlap(a: number[][], b: number[][]): boolean {
  if (a.length < 3 || b.length < 3) return false;
  const bbox = (pts: number[][]) => {
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    return {
      x1: Math.min(...xs),
      y1: Math.min(...ys),
      x2: Math.max(...xs),
      y2: Math.max(...ys),
    };
  };
  const A = bbox(a);
  const B = bbox(b);
  return !(A.x2 < B.x1 || B.x2 < A.x1 || A.y2 < B.y1 || B.y2 < A.y1);
}
