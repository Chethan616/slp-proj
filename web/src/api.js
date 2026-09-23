/* Calls to the FastAPI backend.
 *
 * In development VITE_API_URL is unset and Vite proxies /api to a local server.
 * In production it points at the Cloud Run service.
 */

const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

async function unwrap(res) {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `server returned ${res.status}`)
  }
  return res.json()
}

export async function postVoice(blob) {
  const form = new FormData()
  const ext = (blob.type.split('/')[1] || 'webm').split(';')[0]
  form.append('audio', blob, `speech.${ext}`)
  return unwrap(await fetch(`${BASE}/api/voice`, { method: 'POST', body: form }))
}

export async function postText(text) {
  return unwrap(
    await fetch(`${BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }),
  )
}

export async function getInfo() {
  return unwrap(await fetch(`${BASE}/api/info`))
}

/* Cloud Run scales to zero, so the first request after an idle period has to
 * wait for a container to start and load ~700 MB of models. Ping /health on
 * mount so that happens while the visitor is still reading the page. */
export async function wake() {
  const res = await fetch(`${BASE}/health`)
  return unwrap(res)
}
