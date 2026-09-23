import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/* voice-glow is taken from the registry rather than aliased at the Libraries.dev
 * checkout: that folder is deliberately not in the repository, so aliasing it
 * builds locally and then fails on any clean clone. The published package
 * exposes the same `mobile` VoiceBeam preset (`VoiceBeamType` is
 * 'default' | 'pill' | 'mobile'), so nothing is lost by using it. */
export default defineConfig({
  plugins: [react()],
  server: {
    // In development the API runs locally; in production VITE_API_URL points at
    // the deployed backend and requests go there directly (CORS is enabled).
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
    },
  },
})
