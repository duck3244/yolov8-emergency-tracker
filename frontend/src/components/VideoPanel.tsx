import { useEffect, useRef, useState } from 'react';
import { streamUrl } from '@/api/client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function VideoPanel() {
  const [retryKey, setRetryKey] = useState(0);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // MJPEG endpoint disconnect 시 강제 재로드
  function handleError() {
    setTimeout(() => setRetryKey((k) => k + 1), 1500);
  }

  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    el.src = `${streamUrl()}?_t=${retryKey}`;
  }, [retryKey]);

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="text-base">Live Stream</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="aspect-video w-full bg-black flex items-center justify-center">
          <img
            ref={imgRef}
            alt="Live MJPEG"
            onError={handleError}
            className="max-w-full max-h-full object-contain"
          />
        </div>
      </CardContent>
    </Card>
  );
}
