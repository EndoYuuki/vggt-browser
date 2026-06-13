import { useEffect } from "react";
import { fetchCameras, fetchModels, fetchPoints } from "./api/client";
import { useJobSocket } from "./hooks/useJobSocket";
import { useJobStore } from "./state/jobStore";
import { Viewer } from "./components/Viewer/Canvas";
import { ModelSelect } from "./components/ModelSelect";
import { UploadPanel } from "./components/UploadPanel";
import { JobProgress } from "./components/JobProgress";
import { ViewToggles } from "./components/ViewToggles";
import { FrameInspector } from "./components/Panels/FrameInspector";

export default function App() {
  const setModels = useJobStore((s) => s.setModels);
  const jobId = useJobStore((s) => s.jobId);
  const jobState = useJobStore((s) => s.jobState);
  const setResults = useJobStore((s) => s.setResults);

  useJobSocket(jobId);

  useEffect(() => {
    fetchModels()
      .then((r) => setModels(r.models, r.default))
      .catch(() => {});
  }, [setModels]);

  // When a job finishes, fetch the binary points + cameras.
  useEffect(() => {
    if (!jobId || jobState?.status !== "done") return;
    let cancelled = false;
    Promise.all([fetchPoints(jobId), fetchCameras(jobId)])
      .then(([points, cameras]) => {
        if (!cancelled) setResults(points, cameras);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [jobId, jobState?.status, setResults]);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>VGGT Browser</h1>
        <div className="panel">
          <ModelSelect />
        </div>
        <UploadPanel />
        <JobProgress />
        <ViewToggles />
        <FrameInspector />
      </aside>
      <main className="stage">
        <Viewer />
      </main>
    </div>
  );
}
