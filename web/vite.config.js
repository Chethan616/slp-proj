import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Use the checked-in Libraries.dev implementation, including its exact
      // mobile VoiceBeam preset, instead of the registry bundle.
      'voice-glow': path.resolve(import.meta.dirname, '../Libraries.dev-main/packages/voice-glow/src/index.ts'),
      react: path.resolve(import.meta.dirname, 'node_modules/react'),
      'react-dom': path.resolve(import.meta.dirname, 'node_modules/react-dom'),
    },
  },
  server: {
    // In development the API runs locally; in production VITE_API_URL points at
    // Cloud Run and requests go there directly (CORS is enabled on the API).
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
    },
  },
})
