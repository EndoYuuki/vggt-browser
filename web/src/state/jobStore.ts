import { create } from "zustand";
import type {
  CamerasResponse,
  ColorMode,
  JobState,
  ModelInfo,
  PointData,
} from "../types";

interface ViewState {
  showPoints: boolean;
  showCameras: boolean;
  colorMode: ColorMode;
  confThreshold: number; // 0..1, client-side shader filter
  pointSize: number;
  selectedFrame: number | null;
}

interface JobStore {
  // models
  models: ModelInfo[];
  selectedModel: string | null;
  setModels: (models: ModelInfo[], def: string) => void;
  selectModel: (name: string) => void;

  // job
  jobId: string | null;
  jobState: JobState | null;
  setJob: (jobId: string) => void;
  setJobState: (s: JobState) => void;
  resetJob: () => void;

  // results
  points: PointData | null;
  cameras: CamerasResponse | null;
  setResults: (points: PointData, cameras: CamerasResponse) => void;

  // view
  view: ViewState;
  setView: (patch: Partial<ViewState>) => void;

  selectedModelInfo: () => ModelInfo | null;
}

export const useJobStore = create<JobStore>((set, get) => ({
  models: [],
  selectedModel: null,
  setModels: (models, def) =>
    set({ models, selectedModel: get().selectedModel ?? def }),
  selectModel: (name) => set({ selectedModel: name }),

  jobId: null,
  jobState: null,
  setJob: (jobId) => set({ jobId, jobState: null, points: null, cameras: null }),
  setJobState: (s) => set({ jobState: s }),
  resetJob: () =>
    set({ jobId: null, jobState: null, points: null, cameras: null }),

  points: null,
  cameras: null,
  setResults: (points, cameras) =>
    set({
      points,
      cameras,
      view: { ...get().view, selectedFrame: null },
    }),

  view: {
    showPoints: true,
    showCameras: true,
    colorMode: "rgb",
    confThreshold: 0,
    pointSize: 2,
    selectedFrame: null,
  },
  setView: (patch) => set({ view: { ...get().view, ...patch } }),

  selectedModelInfo: () => {
    const { models, selectedModel } = get();
    return models.find((m) => m.name === selectedModel) ?? null;
  },
}));
