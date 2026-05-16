import { useEffect, useRef } from 'react';

/** Web Audio API로 짧은 경보음을 재생한다. 외부 음원 파일 없이 동작. */
export function useAlertSound(enabled: boolean, severity: 'normal' | 'warning' | 'emergency') {
  const ctxRef = useRef<AudioContext | null>(null);
  const lastPlayedRef = useRef<number>(0);
  const prevSeverityRef = useRef<typeof severity>('normal');

  useEffect(() => {
    if (!enabled) return;
    // severity가 normal → warning/emergency 로 전이될 때만 1회 재생 (또는 5초 throttle)
    const wasElevated = prevSeverityRef.current !== 'normal';
    const isElevated = severity !== 'normal';
    prevSeverityRef.current = severity;
    if (!isElevated) return;

    const now = Date.now();
    if (wasElevated && now - lastPlayedRef.current < 5000) return;
    lastPlayedRef.current = now;

    try {
      if (!ctxRef.current) {
        const Ctx = window.AudioContext ?? (window as any).webkitAudioContext;
        if (!Ctx) return;
        ctxRef.current = new Ctx();
      }
      const ctx = ctxRef.current;
      const beepCount = severity === 'emergency' ? 3 : 1;
      const baseFreq = severity === 'emergency' ? 880 : 660;
      for (let i = 0; i < beepCount; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'square';
        osc.frequency.value = baseFreq;
        const start = ctx.currentTime + i * 0.25;
        const end = start + 0.18;
        gain.gain.setValueAtTime(0.0001, start);
        gain.gain.exponentialRampToValueAtTime(0.2, start + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, end);
        osc.connect(gain).connect(ctx.destination);
        osc.start(start);
        osc.stop(end + 0.02);
      }
    } catch {
      /* ignore audio errors */
    }
  }, [enabled, severity]);
}
