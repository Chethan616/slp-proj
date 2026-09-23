import { useEffect, useRef, useState } from 'react'
import { MetalFx } from 'metal-fx'
import { VoiceBeam, useMicrophone } from 'voice-glow'

/* The input bar, wrapped in VoiceBeam so the glow along its bottom edge reacts
 * to the microphone while recording and gathers into a travelling beam while
 * the server is working.
 *
 * The MediaStream from useMicrophone drives both the glow and the MediaRecorder
 * that actually captures the audio sent to Whisper, so there is only ever one
 * microphone permission and one stream.
 */
export default function Composer({ busy, processing, onVoice, onText, onError, info, speak, setSpeak }) {
  const mic = useMicrophone()
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [text, setText] = useState('')

  const [live, setLive] = useState('')

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const startedAtRef = useRef(0)
  const areaRef = useRef(null)
  const chipRef = useRef(null)
  const sttRef = useRef(null)

  useEffect(() => () => {
    clearInterval(timerRef.current)
    stopLivePreview()
  }, [])

  // Grow the textarea with its content, up to the CSS max-height.
  useEffect(() => {
    const el = areaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text])

  /* A live preview of what is being said, so the field is not blank while you
   * talk. This is the browser's own SpeechRecognition, which returns interim
   * results as you speak — Whisper cannot, because it transcribes a complete
   * recording in one pass on the server.
   *
   * It is strictly a preview: the transcript that reaches the classifier, and
   * the one shown in the conversation, always comes from Whisper. Where the API
   * is missing (Firefox, Safari) the preview is simply absent and nothing else
   * changes. */
  function startLivePreview() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) return
    try {
      const rec = new SR()
      rec.continuous = true
      rec.interimResults = true
      rec.lang = 'en-US'
      rec.onresult = (e) => {
        let text = ''
        for (let i = 0; i < e.results.length; i += 1) text += e.results[i][0].transcript
        setLive(text.trim())
      }
      rec.onerror = () => {}
      rec.start()
      sttRef.current = rec
    } catch {
      // A preview failing must never take the recording down with it.
      sttRef.current = null
    }
  }

  function stopLivePreview() {
    try {
      sttRef.current?.stop()
    } catch {
      /* already stopped */
    }
    sttRef.current = null
  }

  async function startRecording() {
    const stream = await mic.start()
    if (!stream) {
      onError(
        mic.state === 'denied'
          ? 'Microphone access was blocked. Allow the mic in your browser, or type instead.'
          : 'No microphone available — type your question instead.',
      )
      return
    }

    chunksRef.current = []
    const recorder = new MediaRecorder(stream)
    recorderRef.current = recorder
    recorder.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data)
    recorder.onstop = handleStop
    recorder.start()

    setLive('')
    startLivePreview()

    startedAtRef.current = Date.now()
    setElapsed(0)
    setRecording(true)
    timerRef.current = setInterval(() => {
      const secs = (Date.now() - startedAtRef.current) / 1000
      setElapsed(secs)
      // Hard stop so a forgotten open mic cannot upload a huge file.
      if (secs > 30) stopRecording()
    }, 100)
  }

  function stopRecording() {
    const rec = recorderRef.current
    if (rec && rec.state === 'recording') rec.stop()
  }

  function handleStop() {
    const seconds = (Date.now() - startedAtRef.current) / 1000
    const type = recorderRef.current?.mimeType || 'audio/webm'
    clearInterval(timerRef.current)
    stopLivePreview()
    setRecording(false)
    setElapsed(0)
    mic.stop()
    setLive('')

    const blob = new Blob(chunksRef.current, { type })
    if (seconds < 0.4 || blob.size < 1200) {
      onError('That recording was too short — hold on a moment longer and speak clearly.')
      return
    }
    onVoice(blob)
  }

  function submitText(e) {
    e?.preventDefault()
    const value = text.trim()
    if (!value || busy) return
    setText('')
    onText(value)
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) submitText(e)
  }

  const canSend = text.trim().length > 0 && !busy

  return (
    <div className="composer-wrap">
      <VoiceBeam
        stream={recording ? mic.stream : null}
        processing={processing}
        colorVariant="colorful"
        theme="dark"
        type="default"
      >
        <div className="composer">
          <textarea
            ref={areaRef}
            rows={1}
            maxLength={500}
            value={recording ? live : text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={recording ? 'Listening…' : 'Ask me anything..'}
            disabled={busy}
            readOnly={recording}
            className={recording ? 'live' : undefined}
          />

          <div className="composer-bar">
            <span className="model-chip" ref={chipRef}>
              <span className={`dot${info ? '' : ' off'}`} />
              {info ? 'DistilBERT · 41 intents' : 'connecting…'}
            </span>

            {recording && <span className="timer">{elapsed.toFixed(1)}s</span>}

            <span className="spacer" />

            <button
              className={`round-btn${recording ? ' recording' : ''}`}
              onClick={() => (recording ? stopRecording() : startRecording())}
              disabled={busy || !mic.supported}
              aria-label={recording ? 'Stop recording' : 'Start recording'}
              title={mic.supported ? 'Record a question' : 'Microphone not supported'}
            >
              {recording ? (
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true">
                  <path fill="currentColor" d="M12 14.5a3 3 0 0 0 3-3v-5a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Z" />
                  <path fill="currentColor" d="M17.8 11.3a.8.8 0 0 0-1.6 0 4.2 4.2 0 0 1-8.4 0 .8.8 0 0 0-1.6 0 5.8 5.8 0 0 0 5 5.7v2.2a.8.8 0 0 0 1.6 0V17a5.8 5.8 0 0 0 5-5.7Z" />
                </svg>
              )}
            </button>

            {/* The metal shader paints over its host element, so the send
                button keeps its own markup and MetalFx wraps it. Reflection
                targets let the model chip show up in the metal. */}
            <MetalFx
              preset="chromatic"
              variant="circle"
              theme="dark"
              innerShadow
              reflectionTargets={[chipRef]}
              strength={canSend ? 1 : 0.45}
            >
              <button
                type="button"
                className="metal-circle"
                onClick={submitText}
                disabled={!canSend}
                aria-label="Send message"
              >
                <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
                  <path
                    fill="none" stroke="currentColor" strokeWidth="2.2"
                    strokeLinecap="round" strokeLinejoin="round"
                    d="M12 19V5m0 0-6 6m6-6 6 6"
                  />
                </svg>
              </button>
            </MetalFx>
          </div>
        </div>
      </VoiceBeam>

      <div className="composer-foot">
        <label className="speak-toggle">
          <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
          Read replies aloud
        </label>
        <span>
          {info
            ? `${info.stt_model} → ${info.intent_model}`
            : 'waking the model server…'}
        </span>
        <a href="https://github.com/Chethan616/slp-proj" target="_blank" rel="noreferrer">
          source
        </a>
      </div>
    </div>
  )
}
