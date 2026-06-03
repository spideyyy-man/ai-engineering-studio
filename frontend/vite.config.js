var _a;
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    server: {
        port: 5180,
        strictPort: true,
        proxy: {
            "/api": {
                target: (_a = process.env.VITE_API_URL) !== null && _a !== void 0 ? _a : "http://localhost:8000",
                changeOrigin: true,
            },
        },
    },
    preview: {
        port: 5180,
        strictPort: true,
    },
});
