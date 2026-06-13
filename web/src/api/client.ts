import type {
  CamerasResponse,
  JobState,
  ModelsResponse,
  PointData,
} from "../types";

export async function fetchModels(): Promise<ModelsResponse> {
  const r = await fetch("/api/models");
  if (!r.ok) throw new Error("failed to fetch models");
  return r.json();
}

export interface CreateJobParams {
  files: File[];
  model: string;
  frames?: number;
  resolution?: number;
  fps?: number;
  confThreshold?: number;
  pointSource?: "depth" | "pointmap";
}

export async function createJob(p: CreateJobParams): Promise<{ job_id: string }> {
  const fd = new FormData();
  for (const f of p.files) fd.append("files", f);
  fd.append("model", p.model);
  if (p.frames != null) fd.append("frames", String(p.frames));
  if (p.resolution != null) fd.append("resolution", String(p.resolution));
  if (p.fps != null) fd.append("fps", String(p.fps));
  if (p.confThreshold != null) fd.append("conf_threshold", String(p.confThreshold));
  if (p.pointSource != null) fd.append("point_source", p.pointSource);
  const r = await fetch("/api/jobs", { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail ?? "failed to create job");
  return r.json();
}

export async function fetchJobState(jobId: string): Promise<JobState> {
  const r = await fetch(`/api/jobs/${jobId}`);
  if (!r.ok) throw new Error("failed to fetch job state");
  return r.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
}

// ---- binary point parser (mirrors shared/wire_format.py VGGB container) ----
const MAGIC = 0x56474742; // "VGGB" big-endian read below as bytes

export async function fetchPoints(jobId: string): Promise<PointData> {
  const r = await fetch(`/api/jobs/${jobId}/points`);
  if (!r.ok) throw new Error("failed to fetch points");
  const buf = await r.arrayBuffer();
  const dv = new DataView(buf);

  // header: magic(4 bytes ascii), version u16, reserved u16, N u32, 4 offsets u32 (LE)
  const m0 = dv.getUint8(0),
    m1 = dv.getUint8(1),
    m2 = dv.getUint8(2),
    m3 = dv.getUint8(3);
  const magic = (m0 << 24) | (m1 << 16) | (m2 << 8) | m3;
  if (magic !== MAGIC) throw new Error("bad point blob magic");

  const n = dv.getUint32(8, true);
  const posOff = dv.getUint32(12, true);
  const rgbOff = dv.getUint32(16, true);
  const confOff = dv.getUint32(20, true);
  const frameOff = dv.getUint32(24, true);

  return {
    count: n,
    positions: new Float32Array(buf.slice(posOff, posOff + n * 3 * 4)),
    colors: new Uint8Array(buf.slice(rgbOff, rgbOff + n * 3)),
    conf: new Float32Array(buf.slice(confOff, confOff + n * 4)),
    frameIdx: new Uint16Array(buf.slice(frameOff, frameOff + n * 2)),
  };
}

export async function fetchCameras(jobId: string): Promise<CamerasResponse> {
  const r = await fetch(`/api/jobs/${jobId}/cameras`);
  if (!r.ok) throw new Error("failed to fetch cameras");
  return r.json();
}

export function depthUrl(jobId: string, frame: number): string {
  return `/api/jobs/${jobId}/depth/${frame}`;
}
export function confUrl(jobId: string, frame: number): string {
  return `/api/jobs/${jobId}/conf/${frame}`;
}
export function glbUrl(jobId: string): string {
  return `/api/jobs/${jobId}/glb`;
}
