import { cancelJob, glbUrl } from "../api/client";
import { useJobStore } from "../state/jobStore";

const TERMINAL = new Set(["done", "failed", "failed_oom", "cancelled"]);

export function JobProgress() {
  const jobId = useJobStore((s) => s.jobId);
  const state = useJobStore((s) => s.jobState);
  if (!jobId) return null;

  const status = state?.status ?? "queued";
  const pct = Math.round((state?.progress ?? 0) * 100);
  const running = !TERMINAL.has(status);

  return (
    <div className="panel">
      <h3>Job</h3>
      <div className="progressbar">
        <div
          className={`fill ${status}`}
          style={{ width: `${status === "done" ? 100 : pct}%` }}
        />
      </div>
      <small>
        {status}
        {state?.stage ? ` · ${state.stage}` : ""} ({pct}%)
      </small>
      {state?.point_count != null && (
        <small>
          {state.point_count.toLocaleString()} pts · {state.frame_count} frames
        </small>
      )}
      {state?.error && <p className="error">{state.error}</p>}
      {status === "failed_oom" && (
        <p className="error">
          GPU ran out of memory. Reduce frame count or resolution and retry.
        </p>
      )}
      {running && (
        <button onClick={() => cancelJob(jobId)} className="secondary">
          Cancel
        </button>
      )}
      {status === "done" && (
        <a className="button" href={glbUrl(jobId)} download>
          Download GLB
        </a>
      )}
    </div>
  );
}
