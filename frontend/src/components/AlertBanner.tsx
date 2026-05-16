import { Volume2, VolumeX, AlertTriangle, AlertCircle } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import type { AlertStatus } from '@/api/types';

interface Props {
  status: AlertStatus;
  inside: number;
  soundEnabled: boolean;
  onSoundToggle: (v: boolean) => void;
}

export function AlertBanner({ status, inside, soundEnabled, onSoundToggle }: Props) {
  if (status === 'normal') {
    return (
      <div className="flex items-center justify-between rounded-md border bg-card px-3 py-2 text-sm text-muted-foreground">
        <span>System normal · {inside} inside</span>
        <SoundToggle on={soundEnabled} onChange={onSoundToggle} />
      </div>
    );
  }
  const isEmergency = status === 'emergency';
  const wrapperCls = isEmergency
    ? 'border-destructive bg-destructive/10 text-destructive'
    : 'border-amber-500 bg-amber-50 text-amber-900 dark:bg-amber-900/30 dark:text-amber-100';
  const Icon = isEmergency ? AlertCircle : AlertTriangle;
  return (
    <div
      role="alert"
      className={`flex items-center justify-between rounded-md border px-3 py-2 ${wrapperCls} ${isEmergency ? 'animate-pulse' : ''}`}
    >
      <div className="flex items-center gap-2 font-medium">
        <Icon className="h-5 w-5" />
        {isEmergency ? 'EMERGENCY' : 'WARNING'} — {inside} inside
      </div>
      <SoundToggle on={soundEnabled} onChange={onSoundToggle} />
    </div>
  );
}

function SoundToggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {on ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
      <Switch checked={on} onCheckedChange={onChange} />
    </div>
  );
}
