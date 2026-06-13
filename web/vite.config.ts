import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `vite dev`, proxy api + ws to the api container.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
