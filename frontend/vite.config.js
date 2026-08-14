import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server is pinned to 5173 to match the backend's CORS allow_origins
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
