import { useEffect, useState } from 'react';
import { api } from '@/api/client';
import type { ConfigPatch, ConfigSummary } from '@/api/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Bell } from 'lucide-react';
import { useDebouncedCallback } from '@/hooks/useDebouncedCallback';

interface Props {
  onChange?: (cfg: ConfigSummary) => void;
}

export function SettingsForm({ onChange }: Props) {
  const [cfg, setCfg] = useState<ConfigSummary | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contactsText, setContactsText] = useState('');

  useEffect(() => {
    api
      .config()
      .then((c) => {
        setCfg(c);
        setContactsText(c.emergency_contacts.join(', '));
      })
      .catch((e) => setError(String(e)));
  }, []);

  const pushPatch = useDebouncedCallback(async (patch: ConfigPatch) => {
    setSaving(true);
    setError(null);
    try {
      const next = await api.patchConfig(patch);
      setCfg(next);
      onChange?.(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }, 350);

  function update<K extends keyof ConfigPatch>(key: K, value: ConfigPatch[K]) {
    if (!cfg) return;
    setCfg({ ...cfg, [key]: value as ConfigSummary[keyof ConfigSummary] });
    pushPatch({ [key]: value } as ConfigPatch);
  }

  if (!cfg) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading…</p>
          {error && <p className="text-xs text-destructive mt-2">{error}</p>}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">Settings</CardTitle>
        <span className="text-xs text-muted-foreground">{saving ? 'saving…' : 'saved'}</span>
      </CardHeader>
      <CardContent className="space-y-5">
        <Section title="Thresholds">
          <SliderRow
            label="Warning"
            value={cfg.warning_threshold}
            min={1}
            max={200}
            step={1}
            onChange={(v) => update('warning_threshold', v)}
          />
          <SliderRow
            label="Overcrowding"
            value={cfg.overcrowding_threshold}
            min={1}
            max={500}
            step={1}
            onChange={(v) => update('overcrowding_threshold', v)}
          />
          <SliderRow
            label="Notification interval (s)"
            value={cfg.notification_interval}
            min={10}
            max={3600}
            step={10}
            onChange={(v) => update('notification_interval', v)}
          />
        </Section>

        <Section title="Detection">
          <SliderRow
            label="Confidence"
            value={cfg.confidence_threshold}
            min={0.1}
            max={0.95}
            step={0.05}
            format={(v) => v.toFixed(2)}
            onChange={(v) => update('confidence_threshold', v)}
          />
          <SliderRow
            label="IoU"
            value={cfg.iou_threshold}
            min={0.1}
            max={0.95}
            step={0.05}
            format={(v) => v.toFixed(2)}
            onChange={(v) => update('iou_threshold', v)}
          />
          <SliderRow
            label="Frame skip"
            value={cfg.frame_skip}
            min={1}
            max={10}
            step={1}
            onChange={(v) => update('frame_skip', v)}
          />
        </Section>

        <Section title="Tracking">
          <SliderRow
            label="Distance threshold (px)"
            value={cfg.distance_threshold}
            min={5}
            max={200}
            step={1}
            onChange={(v) => update('distance_threshold', v)}
          />
          <SliderRow
            label="Max disappeared (frames)"
            value={cfg.max_disappeared}
            min={1}
            max={120}
            step={1}
            onChange={(v) => update('max_disappeared', v)}
          />
          <SliderRow
            label="Min residence (s)"
            value={cfg.min_residence_time}
            min={0}
            max={5}
            step={0.1}
            format={(v) => v.toFixed(1)}
            onChange={(v) => update('min_residence_time', v)}
          />
        </Section>

        <Section title="Email alerts">
          <div className="flex items-center justify-between">
            <Label htmlFor="enable_email">Enable email</Label>
            <Switch
              id="enable_email"
              checked={cfg.enable_email}
              onCheckedChange={(v) => update('enable_email', v)}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="contacts">Recipients (comma-separated)</Label>
            <Input
              id="contacts"
              value={contactsText}
              onChange={(e) => setContactsText(e.target.value)}
              onBlur={() =>
                update(
                  'emergency_contacts',
                  contactsText
                    .split(',')
                    .map((s) => s.trim())
                    .filter(Boolean)
                )
              }
              placeholder="emergency@example.com, ops@example.com"
            />
            <p className="text-xs text-muted-foreground">
              SMTP credentials are loaded from <code>.env</code>.
              {cfg.email_configured ? ' Credentials detected.' : ' No password configured.'}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => api.sendTestAlert().catch((e) => setError(String(e)))}
          >
            <Bell className="h-4 w-4" /> Send test alert
          </Button>
        </Section>

        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3 border-t pt-4 first:border-t-0 first:pt-0">
      <h4 className="text-sm font-semibold text-muted-foreground">{title}</h4>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}

function SliderRow({ label, value, min, max, step, onChange, format }: SliderRowProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span>{label}</span>
        <span className="font-mono text-xs text-muted-foreground">
          {format ? format(value) : value}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0])}
      />
    </div>
  );
}
