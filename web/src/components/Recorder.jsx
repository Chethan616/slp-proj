import { useEffect, useRef, useState } from 'react'

/* Microphone capture via MediaRecorder, with a live input-level ring so the
 * user can see that audio is actually reaching the page. */
export default function Recorder({ disabled, onRecorded, onError }) {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const audioCtxRef = useRef(null)
  const rafRef = useRef(null)
  const timerRef = useRef(null)
  const startedAtRef = useRef(0)
  const levelRef = useRef(null)

  // Release the microphone if the component goes away mid-recording.
  useEffect(() => () => teardown(), [])

  function teardown() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    if (timerRef.current) clearInterval(timerRef.current)
    rafRef.current = null
    timerRef.current = null
    if (audioCtxRef.current) audioCtxRef.current.close().catch(() => {})
    audioCtxRef.current = null
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (levelRef.current) levelRef.current.style.transform = 'scale(1)'
  }

  function drawLevel(analyser) {
    const data = new Uint8Array(analyser.frequencyBinCount)
    const tick = () => {
      analyser.getByteTimeDomainData(data)
      let peak = 0
      for (const v of data) peak = Math.max(peak, Math.abs(v - 128) / 128)
      if (levelRef.current) {
        levelRef.current.style.transform = `scale(${1 + Math.min(peak, 1) * 0.45})`
        levelRef.current.style.opacity = String(0.35 + Math.min(peak, 1) * 0.65)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    tick()
  }

  async function start() {
    if (disabled) return
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      })
    } catch {
      onError('Microphone access was blocked. Allow the mic in your browser, or type your message below.')
      return
    }

    streamRef.current = stream
    chunksRef.current = []

    const recorder = new MediaRecorder(stream)
    recorderRef.current = recorder
    recorder.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data)
    recorder.onstop = handleStop
    recorder.start()

    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    audioCtxRef.current = ctx
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    ctx.createMediaStreamSource(stream).connect(analyser)
    drawLevel(analyser)

    startedAtRef.current = Date.now()
    setElapsed(0)
    timerRef.current = setInterval(() => {
      const secs = (Date.now() - startedAtRef.current) / 1000
      setElapsed(secs)
      // Hard stop so a forgotten open mic cannot upload a huge file.
      if (secs > 30) stop()
    }, 100)

    setRecording(true)
  }

  function stop() {
    const rec = recorderRef.current
    if (rec && rec.state === 'recording') rec.stop()
  }

  function handleStop() {
    const seconds = (Date.now() - startedAtRef.current) / 1000
    const type = recorderRef.current?.mimeType || 'audio/webm'
    teardown()
    setRecording(false)
    setElapsed(0)

    const blob = new Blob(chunksRef.current, { type })
    if (seconds < 0.4 || blob.size < 1200) {
      onError('That recording was too short — hold on a moment longer and speak clearly.')
      return
    }
    onRecorded(blob)
  }

  // Space bar as push-to-talk, unless focus is in a text field.
  useEffect(() => {
    const isTyping = () => ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)
    const down = (e) => {
      if (e.code !== 'Space' || e.repeat || isTyping()) return
      e.preventDefault()
      if (!recording) start()
    }
    const up = (e) => {
      if (e.code !== 'Space' || isTyping()) return
      stop()
    }
    document.addEventListener('keydown', down)
    document.addEventListener('keyup', up)
    return () => {
      document.removeEventListener('keydown', down)
      document.removeEventListener('keyup', up)
    }
  })

  const supported = !!(navigator.mediaDevices && window.MediaRecorder)

  return (
    <div className="mic-row">
      <button
        className={`mic${recording ? ' recording' : ''}`}
        onClick={() => (recording ? stop() : start())}
        disabled={disabled || !supported}
        aria-label={recording ? 'Stop recording' : 'Start recording'}
      >
        <svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true">
          <path fill="currentColor" d="M12 15a3.5 3.5 0 0 0 3.5-3.5v-6a3.5 3.5 0 1 0-7 0v6A3.5 3.5 0 0 0 12 15Z" />
          <path fill="currentColor" d="M18.5 11.5a.9.9 0 0 0-1.8 0 4.7 4.7 0 0 1-9.4 0 .9.9 0 0 0-1.8 0 6.5 6.5 0 0 0 5.6 6.4V21a.9.9 0 0 0 1.8 0v-3.1a6.5 6.5 0 0 0 5.6-6.4Z" />
        </svg>
        <span className="level" ref={levelRef} />
      </button>
      <div className="mic-meta">
        <span>
          {!supported
            ? 'Microphone not supported — use the text box'
            : disabled
              ? 'Processing…'
              : recording
                ? 'Listening — click to stop'
                : 'Click to speak'}
        </span>
        {recording && <span className="timer">{elapsed.toFixed(1)}s</span>}
      </div>
    </div>
  )
}
