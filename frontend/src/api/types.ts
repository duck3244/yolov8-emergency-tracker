// 백엔드 Pydantic 스키마와 동기 유지. `npm run gen:api`로 OpenAPI에서 자동 생성도 가능.

export type AlertStatus = 'normal' | 'warning' | 'emergency';

export interface Counts {
  entered: number;
  exited: number;
  current_inside: number;
  area_name: string;
  max_inside_seen: number;
}

export interface State {
  running: boolean;
  source: string | null;
  session_id: string | null;
  frame_count: number;
  fps: number;
  counts: Counts;
  alert_status: AlertStatus;
  last_error: string | null;
}

export interface HealthResponse {
  status: string;
  video_running: boolean;
  source: string | null;
  session_id: string | null;
  device: string | null;
  uptime_seconds: number;
  model_ready: boolean;
}

export interface SourceItem {
  id: string;
  label: string;
  kind: 'webcam' | 'file';
  size_bytes?: number;
}

export interface ConfigSummary {
  confidence_threshold: number;
  iou_threshold: number;
  device: string;
  frame_skip: number;
  frame_width: number;
  frame_height: number;
  distance_threshold: number;
  max_disappeared: number;
  min_residence_time: number;
  area_name: string;
  overcrowding_threshold: number;
  warning_threshold: number;
  notification_interval: number;
  enable_email: boolean;
  emergency_contacts: string[];
  email_configured: boolean;
  location_name: string;
}

export type ConfigPatch = Partial<{
  confidence_threshold: number;
  iou_threshold: number;
  device: string;
  frame_skip: number;
  distance_threshold: number;
  max_disappeared: number;
  overcrowding_threshold: number;
  warning_threshold: number;
  notification_interval: number;
  enable_email: boolean;
  emergency_contacts: string[];
  min_residence_time: number;
}>;

export interface Areas {
  entrance: number[][];
  exit: number[][];
}
