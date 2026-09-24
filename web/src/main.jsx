import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

/* No StrictMode.
 *
 * StrictMode deliberately mounts, unmounts and remounts every component once in
 * development. The canvas libraries this UI is built on (metal-fx, voice-glow,
 * bot-avatars, thinking-orbs) each attach to a shared render loop on mount and
 * tear it down on unmount; the teardown from that throwaway first mount stops
 * the loop and the remount does not restart it. The visible symptom is metal
 * and glow that paint one frame and then freeze - in development only, since
 * StrictMode does not double-mount in a production build.
 *
 * Verified: with StrictMode the metal canvas produced 1 unique frame across 5
 * samples; without it, 5 of 5, matching the deployed build.
 */
createRoot(document.getElementById('root')).render(<App />)
