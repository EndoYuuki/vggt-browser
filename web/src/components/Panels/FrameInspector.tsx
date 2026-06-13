import { confUrl, depthUrl } from "../../api/client";
import { useJobStore } from "../../state/jobStore";

/** 2D depth + confidence images for the selected frame (lazy-loaded PNGs).
 * Lives outside the WebGL canvas — these are 2D images, not 3D geometry. */
export function FrameInspector() {
  const jobId = useJobStore((s) => s.jobId);
  const cameras = useJobStore((s) => s.cameras);
  const view = useJobStore((s) => s.view);
  const setView = useJobStore((s) => s.setView);

  if (!jobId || !cameras) return null;
  const frame = view.selectedFrame;

  return (
    <div className="panel">
      <h3>Frame inspector</h3>
      <div className="field">
        <span>
          Frame: {frame ?? "none"} (click a camera frustum, or pick below)
        </span>
        <select
          value={frame ?? ""}
          onChange={(e) =>
            setView({
              selectedFrame: e.target.value === "" ? null : Number(e.target.value),
            })
          }
        >
          <option value="">— none —</option>
          {cameras.cameras.map((c) => (
            <option key={c.frame} value={c.frame}>
              frame {c.frame}
            </option>
          ))}
        </select>
      </div>
      {frame != null && (
        <div className="frame-images">
          <figure>
            <img src={depthUrl(jobId, frame)} alt={`depth ${frame}`} loading="lazy" />
            <figcaption>Depth</figcaption>
          </figure>
          <figure>
            <img src={confUrl(jobId, frame)} alt={`conf ${frame}`} loading="lazy" />
            <figcaption>Confidence</figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}
