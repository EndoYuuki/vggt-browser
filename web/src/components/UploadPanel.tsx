import { useState } from "react";
import { createJob } from "../api/client";
import { useJobStore } from "../state/jobStore";
import { CapControls } from "./CapControls";

const VIDEO_RE = /\.(mp4|mov|avi|mkv|webm)$/i;

export function UploadPanel() {
  const selectedModel = useJobStore((s) => s.selectedModel);
  const setJob = useJobStore((s) => s.setJob);
  const [files, setFiles] = useState<File[]>([]);
  const [frames, setFrames] = useState(0);
  const [resolution, setResolution] = useState(0);
  const [fps, setFps] = useState(2);
  const [confThreshold, setConfThreshold] = useState(0.1);
  const [pointSource, setPointSource] = useState<"depth" | "pointmap">("depth");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasVideo = files.some((f) => VIDEO_RE.test(f.name));

  const submit = async () => {
    if (!selectedModel || files.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const { job_id } = await createJob({
        files,
        model: selectedModel,
        frames: frames || undefined,
        resolution: resolution || undefined,
        fps,
        confThreshold,
        pointSource,
      });
      setJob(job_id);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h3>Input</h3>
      <input
        type="file"
        multiple
        accept="image/*,video/*"
        onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
      />
      {files.length > 0 && (
        <small>
          {hasVideo ? "video" : `${files.length} image(s)`} selected
        </small>
      )}

      <CapControls
        frames={frames}
        setFrames={setFrames}
        resolution={resolution}
        setResolution={setResolution}
        fps={fps}
        setFps={setFps}
        hasVideo={hasVideo}
      />

      <label className="field">
        <span>Point source</span>
        <select
          value={pointSource}
          onChange={(e) => setPointSource(e.target.value as "depth" | "pointmap")}
        >
          <option value="depth">depth + camera (higher accuracy)</option>
          <option value="pointmap">point map (direct)</option>
        </select>
      </label>

      <label className="field">
        <span>Drop lowest-confidence: {Math.round(confThreshold * 100)}%</span>
        <input
          type="range"
          min={0}
          max={0.9}
          step={0.05}
          value={confThreshold}
          onChange={(e) => setConfThreshold(Number(e.target.value))}
        />
      </label>

      <button disabled={busy || !selectedModel || files.length === 0} onClick={submit}>
        {busy ? "Submitting…" : "Reconstruct"}
      </button>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
