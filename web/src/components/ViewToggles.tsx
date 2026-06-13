import { useJobStore } from "../state/jobStore";
import type { ColorMode } from "../types";

const COLOR_MODES: ColorMode[] = ["rgb", "frame", "confidence"];

export function ViewToggles() {
  const points = useJobStore((s) => s.points);
  const view = useJobStore((s) => s.view);
  const setView = useJobStore((s) => s.setView);
  if (!points) return null;

  return (
    <div className="panel">
      <h3>View</h3>
      <label className="check">
        <input
          type="checkbox"
          checked={view.showPoints}
          onChange={(e) => setView({ showPoints: e.target.checked })}
        />
        Points
      </label>
      <label className="check">
        <input
          type="checkbox"
          checked={view.showCameras}
          onChange={(e) => setView({ showCameras: e.target.checked })}
        />
        Cameras
      </label>

      <label className="field">
        <span>Color by</span>
        <select
          value={view.colorMode}
          onChange={(e) => setView({ colorMode: e.target.value as ColorMode })}
        >
          {COLOR_MODES.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Confidence threshold: {view.confThreshold.toFixed(2)}</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.02}
          value={view.confThreshold}
          onChange={(e) => setView({ confThreshold: Number(e.target.value) })}
        />
      </label>

      <label className="field">
        <span>Point size: {view.pointSize.toFixed(1)}</span>
        <input
          type="range"
          min={0.5}
          max={8}
          step={0.5}
          value={view.pointSize}
          onChange={(e) => setView({ pointSize: Number(e.target.value) })}
        />
      </label>
    </div>
  );
}
