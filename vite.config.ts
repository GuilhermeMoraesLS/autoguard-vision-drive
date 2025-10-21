import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const allowedHost = env.VITE_ALLOWED_HOST || "localhost";
  return {
    server: {
      // Allow ngrok domain to reach the dev server
      allowedHosts: [allowedHost],
      host: "::",
      port: 8080,
    },
    preview: {
      // Mirror allowed host for `vite preview` as well
      allowedHosts: [allowedHost],
    },
    plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
});
