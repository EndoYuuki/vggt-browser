export interface ModelCaps {
  max_frames: number;
  default_frames: number;
  recommended_resolution: number;
  max_resolution: number;
}

export interface ModelInfo {
  name: string;
  adapter: string;
  display_name: string;
  description: string;
  caps: ModelCaps;
}

export interface ModelsResponse {
  default: string;
  models: ModelInfo[];
}

export type JobStatus =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "failed_oom"
  | "cancelled";

export interface JobState {
  job_id: string;
  status: JobStatus;
  stage?: string;
  progress?: number;
  error?: string;
  frame_count?: number;
  point_count?: number;
  image_size?: [number, number];
}

export interface PointData {
  count: number;
  positions: Float32Array; // N*3
  colors: Uint8Array; // N*3
  conf: Float32Array; // N
  frameIdx: Uint16Array; // N
}

export interface CameraInfo {
  frame: number;
  extrinsic: number[][]; // 3x4 world->cam
  intrinsic: number[][]; // 3x3
}

export interface CamerasResponse {
  frame_count: number;
  image_size: { height: number; width: number };
  cameras: CameraInfo[];
}

export type ColorMode = "rgb" | "frame" | "confidence";
