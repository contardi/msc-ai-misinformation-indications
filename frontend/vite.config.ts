import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { API_PATHS } from './src/apiPaths'

const proxyTarget = process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000'

const proxy = Object.fromEntries(
  Object.values(API_PATHS).map((path) => [path, { target: proxyTarget, changeOrigin: true }]),
)

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true, // bind 0.0.0.0 so the container's port is reachable from the host
    port: 5173,
    allowedHosts: ['frontend', 'localhost'],
    proxy,
  },
})
