import { useEffect } from "react";
import { useJobStore } from "../state/jobStore";

interface Props {
  frames: number;
  setFrames: (n: number) => void;
  resolution: number;
  setResolution: (n: number) => void;
  fps: number;
  setFps: (n: number) => void;
  hasVideo: boolean;
}

export function CapControls({
  frames,
  setFrames,
  resolution,
  setResolution,
  fps,
  setFps,
  hasVideo,
}: Props) {
  const info = useJobStore((s) => s.selectedModelInfo());

  // Clamp current values to the selected model's caps when it changes.
  useEffect(() => {
    if (!info) return;
    setFrames(Math.min(frames || info.caps.default_frames, info.caps.max_frames));
    setResolution(
      Math.min(resolution || info.caps.recommended_resolution, info.caps.max_resolution),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info?.name]);

  if (!info) return null;
  const caps = info.caps;
  const nearMax = frames >= caps.max_frames * 0.9;

  return (
    <div className="field">
      <label>
        <span>
          Frames: {frames} / {caps.max_frames}
        </span>
        <input
          type="range"
          min={1}
          max={caps.max_frames}
          value={frames}
          onChange={(e) => setFrames(Number(e.target.value))}
        />
        {nearMax && <small className="warn">⚠ near VRAM cap</small>}
      </label>
      <label>
        <span>
          Resolution: {resolution} (max {caps.max_resolution})
        </span>
        <input
          type="range"
          min={128}
          max={caps.max_resolution}
          step={14}
          value={resolution}
          onChange={(e) => setResolution(Number(e.target.value))}
        />
      </label>
      {hasVideo && (
        <label>
          <span>Video sampling: {fps} fps</span>
          <input
            type="range"
            min={0.5}
            max={10}
            step={0.5}
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
          />
        </label>
      )}
    </div>
  );
}
