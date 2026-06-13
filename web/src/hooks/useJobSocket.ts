import { useEffect, useRef } from "react";
import { fetchJobState } from "../api/client";
import { useJobStore } from "../state/jobStore";
import type { JobState } from "../types";

const TERMINAL = new Set(["done", "failed", "failed_oom", "cancelled"]);

/** Subscribe to job progress via WebSocket, with polling fallback. */
export function useJobSocket(jobId: string | null) {
  const setJobState = useJobStore((s) => s.setJobState);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let closed = false;
    let ws: WebSocket | null = null;

    const stopPoll = () => {
      if (pollRef.current != null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };

    const startPoll = () => {
      if (pollRef.current != null) return;
      pollRef.current = window.setInterval(async () => {
        try {
          const s = await fetchJobState(jobId);
          setJobState(s);
          if (TERMINAL.has(s.status)) stopPoll();
        } catch {
          /* keep polling */
        }
      }, 1000);
    };

    try {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${jobId}`);
      ws.onmessage = (ev) => {
        if (closed) return;
        const s = JSON.parse(ev.data) as JobState;
        setJobState(s);
      };
      ws.onerror = () => startPoll();
      ws.onclose = () => {
        if (!closed) startPoll();
      };
    } catch {
      startPoll();
    }

    return () => {
      closed = true;
      stopPoll();
      ws?.close();
    };
  }, [jobId, setJobState]);
}
