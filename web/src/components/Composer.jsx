import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react'
import { MetalFx, useMetalBend } from 'metal-fx'
import { VoiceBeam, useMicrophone } from 'voice-glow'
import { useMediaQuery } from '../useMediaQuery'

function SpeakToggle({ speak, setSpeak, className = '' }) {
  return (
    <button
      type="button"
      className={`aloud${className ? ` ${className}` : ''}${speak ? ' on' : ''}`}
      onClick={() => setSpeak(!speak)}
      aria-pressed={speak}
      aria-label={speak ? 'Mute' : 'Speak'}
    >
      {speak ? (
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
          <path fill="currentColor" d="M4 9v6h4l5 4V5L8 9H4Z" />
          <path
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12"
          />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
          <path fill="currentColor" d="M4 9v6h4l5 4V5L8 9H4Z" />
          <path
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            d="m17 9.5 4 5m0-5-4 5"
          />
        </svg>
      )}

      <span className="mobile-agent-label">
        {speak ? 'Mute' : 'Speak'}
      </span>
    </button>
  )
}

/* The input bar, wrapped in VoiceBeam so the glow along its bottom edge reacts
 * to the microphone while recording and gathers into a travelling beam while
 * the server is working.
 *
 * The MediaStream from useMicrophone drives both the glow and the MediaRecorder
 * that actually captures the audio sent to Whisper, so there is only ever one
 * microphone permission and one stream.
 */
export default function Composer({
  busy,
  processing,
  onVoice,
  onText,
  onError,
  info,
  speak,
  setSpeak,
  ready,
  conn,
  onNotReady
}) {
  const mic = useMicrophone()
  const [recording, setRecording] = useState(false)
  /* Only whether the field is empty, not its contents: this flips once when you
   * start typing and once when you clear, rather than on every keystroke. */
  const [hasText, setHasText] = useState(false)

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const startedAtRef = useRef(0)
  const fieldRef = useRef(null)
  const chipRef = useRef(null)
  const sttRef = useRef(null)
  const sendRef = useRef(null)
  const micRef = useRef(null)
  const cancelledRef = useRef(false)

  // voice-glow ships a geometry preset tuned for the bottom of a phone screen.
  const isPhone = useMediaQuery('(max-width: 640px)')

  /* Stable identity: a fresh array each render makes MetalFx tear down and
   * rebuild its reflections on every re-render.
   *
   * The microphone is deliberately not a reflection target. MetalFx paints a
   * reflection canvas on top of each target and repaints it continuously, so
   * putting one over a button that sits above the animating beam made the
   * control look like it was flickering. */
  const reflectTargets = useMemo(() => [chipRef], [])

  // Cursor-driven liquid dent on the send button: the ring stretches and
  // recoils under the pointer instead of sitting there as a static bevel.
  useMetalBend(sendRef)

  useEffect(() => () => {
    clearInterval(timerRef.current)
    stopLivePreview()
  }, [])

  /* A live preview of what is being said, so the field is not blank while you
   * talk. This is the browser's own SpeechRecognition, which returns interim
   * results as you speak - Whisper cannot, because it transcribes a complete
   * recording in one pass on the server.
   *
   * It is strictly a preview: the transcript that reaches the classifier, and
   * the one shown in the conversation, always comes from Whisper. Where the API
   * is missing (Firefox, Safari) the preview is simply absent and nothing else
   * changes.
   */
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

        for (let i = 0; i < e.results.length; i += 1) {
          text += e.results[i][0].transcript
        }

        fieldRef.current?.setLive(text.trim())
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
    if (!ready) {
      onNotReady()
      return
    }

    const stream = await mic.start()

    if (!stream) {
      onError(
        mic.state === 'denied'
          ? 'Microphone access was blocked. Allow the mic in your browser, or type instead.'
          : 'No microphone available - type your question instead.'
      )
      return
    }

    chunksRef.current = []

    const recorder = new MediaRecorder(stream)
    recorderRef.current = recorder

    recorder.ondataavailable = (e) => {
      if (e.data.size) {
        chunksRef.current.push(e.data)
      }
    }

    recorder.onstop = handleStop
    recorder.start()

    fieldRef.current?.setLive('')
    startLivePreview()

    startedAtRef.current = Date.now()
    setRecording(true)

    /* This interval deliberately sets no state. The elapsed reading lives in
     * <RecTimer>, which re-renders itself; if the whole composer re-rendered
     * ten times a second, MetalFx would rebuild its reflections just as often
     * and the controls would visibly flicker. */
    timerRef.current = setInterval(() => {
      const secs = (Date.now() - startedAtRef.current) / 1000

      // Hard stop so a forgotten open mic cannot upload a huge file.
      if (secs > 30) {
        stopRecording()
      }
    }, 250)
  }

  /* Stopping submits; discarding throws the recording away. Both go through
   * MediaRecorder.stop(), since that is the only way to release the device,
   * so a flag tells handleStop which one asked. */
  function stopRecording() {
    cancelledRef.current = false
    const rec = recorderRef.current

    if (rec && rec.state === 'recording') {
      rec.stop()
    }
  }

  function cancelRecording() {
    cancelledRef.current = true
    const rec = recorderRef.current

    if (rec && rec.state === 'recording') {
      rec.stop()
    }
  }

  function handleStop() {
    const seconds = (Date.now() - startedAtRef.current) / 1000
    const type = recorderRef.current?.mimeType || 'audio/webm'

    clearInterval(timerRef.current)
    stopLivePreview()
    setRecording(false)
    mic.stop()
    fieldRef.current?.setLive('')

    // Discarded: release the mic, keep the audio out of the conversation.
    if (cancelledRef.current) {
      cancelledRef.current = false
      return
    }

    const blob = new Blob(chunksRef.current, { type })

    if (seconds < 0.4 || blob.size < 1200) {
      onError(
        'That recording was too short - hold on a moment longer and speak clearly.'
      )
      return
    }

    onVoice(blob)
  }

  function submitText(e) {
    e?.preventDefault()

    if (!ready) {
      onNotReady()
      return
    }

    const value = (fieldRef.current?.getText() ?? '').trim()

    if (!value || busy) return

    fieldRef.current?.clear()
    onText(value)
  }

  function clearComposer() {
    // While recording, the cross discards. Only the square stop submits -
    // stopRecording() would have sent the audio the user just asked to drop.
    if (recording) {
      cancelRecording()
      return
    }

    fieldRef.current?.clear()
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      submitText(e)
    }
  }

  /* Deliberately not gated on `ready`: a disabled button swallows the click,
   * and the point is that pressing it explains the wait. submitText guards. */
  const canSend = hasText && !busy

  return (
    <div className="composer-wrap">
      <VoiceBeam
        stream={recording ? mic.stream : null}
        processing={processing}
        colorVariant="colorful"
        theme="dark"
        type={isPhone ? 'mobile' : 'default'}
        borderRadius={isPhone ? 0 : undefined}
        /* The bloom blur is the expensive part of the effect and phones cannot
         * afford it at full radius: a full-height host ran the page at 7fps,
         * which is what looked like flicker. Tighter halo and single-band
         * lobes on phones only. */
        glowSize={isPhone ? 0.75 : 1}
        bands={!isPhone}
        /* The mobile preset's default reach is 3, sized for a host the height
         * of a phone screen. In a short band that overflows and VoiceBeam's own
         * overflow:hidden cuts it off in a straight line, so bring the bloom
         * back inside the band. */
        reach={isPhone ? 1.3 : undefined}
        spread={isPhone ? 0.52 : undefined}
      >
        <div className="composer">
          <ComposerField
            ref={fieldRef}
            recording={recording}
            busy={busy}
            ready={ready}
            onKeyDown={onKeyDown}
            onEmptyChange={setHasText}
          />

          <div className="composer-bar">
            <span className="model-chip" ref={chipRef}>
              <span className={`dot${info ? '' : ' off'}`} />

              {info ? (
                <>
                  <span className="desktop-model-name">
                    DistilBERT ·{' '}
                  </span>

                  <span className="mobile-model-name">
                    Agent (auto){' '}
                    <svg
                      className="agent-chevron"
                      viewBox="0 0 12 12"
                      width="11"
                      height="11"
                      aria-hidden="true"
                    >
                      <path
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="m3 4.5 3 3 3-3"
                      />
                    </svg>{' '}
                    ·{' '}
                  </span>

                  41 intents
                </>
              ) : (
                'connecting…'
              )}
            </span>

            {!ready ? (
              <span
                className={`conn-pill${conn === 'failed' ? ' failed' : ''}`}
                role="status"
              >
                <span className="dot" />
                {conn === 'failed' ? 'Server unavailable' : 'Connecting…'}
              </span>
            ) : (
            <SpeakToggle
              speak={speak}
              setSpeak={setSpeak}
              className="mobile-aloud"
            />
            )}

            <span className="spacer" />

            {recording && (
              <RecTimer startedAt={startedAtRef} />
            )}

            {recording && (
              <button
                className="icon-circle discard"
                onClick={cancelRecording}
                aria-label="Discard recording"
                title="Discard recording"
              >
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <path
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.1"
                    strokeLinecap="round"
                    d="m7 7 10 10M17 7 7 17"
                  />
                </svg>
              </button>
            )}

            <button
              ref={micRef}
              className={`icon-circle mic${recording ? ' recording' : ''}`}
              onClick={() =>
                recording ? stopRecording() : startRecording()
              }
              disabled={busy || !mic.supported}
              data-waiting={!ready ? 'true' : undefined}
              aria-label={
                recording ? 'Stop recording' : 'Record a question'
              }
              title={
                mic.supported
                  ? recording
                    ? 'Stop recording'
                    : 'Record a question'
                  : 'No microphone available'
              }
            >
              {recording ? (
                <svg
                  viewBox="0 0 24 24"
                  width="15"
                  height="15"
                  aria-hidden="true"
                >
                  <rect
                    x="6"
                    y="6"
                    width="12"
                    height="12"
                    rx="2.5"
                    fill="currentColor"
                  />
                </svg>
              ) : (
                <svg
                  viewBox="0 0 24 24"
                  width="19"
                  height="19"
                  aria-hidden="true"
                >
                  <path
                    fill="currentColor"
                    d="M12 14.5a3 3 0 0 0 3-3v-5a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Z"
                  />
                  <path
                    fill="currentColor"
                    d="M17.8 11.3a.8.8 0 0 0-1.6 0 4.2 4.2 0 0 1-8.4 0 .8.8 0 0 0-1.6 0 5.8 5.8 0 0 0 5 5.7v2.2a.8.8 0 0 0 1.6 0V17a5.8 5.8 0 0 0 5-5.7Z"
                  />
                </svg>
              )}
            </button>

            <button
              type="button"
              className="icon-circle mobile-close"
              onClick={clearComposer}
              disabled={busy}
              aria-label="Clear message"
              title="Clear message"
            >
              <svg
                viewBox="0 0 24 24"
                width="17"
                height="17"
                aria-hidden="true"
              >
                <path
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  d="M6 6l12 12M18 6 6 18"
                />
              </svg>
            </button>

            {!isPhone && (
            <>
            {/* The metal shader paints over its host element, so the send
                button keeps its own markup and MetalFx wraps it. */}
            <MetalFx
              ref={sendRef}
              preset="chromatic"
              variant="circle"
              theme="dark"
              innerShadow
              reflectionTargets={reflectTargets}
              strength={canSend && ready ? 1 : 0.72}
            >
              <button
                type="button"
                className="metal-circle"
                onClick={submitText}
                disabled={!canSend}
                aria-label="Send typed message"
                title="Send typed message"
              >
                <svg
                  viewBox="0 0 24 24"
                  width="17"
                  height="17"
                  aria-hidden="true"
                >
                  <path
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 19V5m0 0-6 6m6-6 6 6"
                  />
                </svg>
              </button>
            </MetalFx>
            </>
            )}
          </div>
        </div>
      </VoiceBeam>

      <div className="composer-foot">
        {/* Duplicate bottom SpeakToggle removed */}

        <span className="foot-models">
          {info
            ? `${info.stt_model} → ${info.intent_model}`
            : 'waking the model server…'}
        </span>
      </div>
    </div>
  )
}

/* Owns its own tick so the elapsed reading updates without re-rendering the
 * composer - and therefore without disturbing the metal. */
function RecTimer({ startedAt }) {
  const [secs, setSecs] = useState(0)

  useEffect(() => {
    const id = setInterval(
      () => setSecs((Date.now() - startedAt.current) / 1000),
      100,
    )
    return () => clearInterval(id)
  }, [startedAt])

  return <span className="rec-timer">{secs.toFixed(1)}s</span>
}

/* Holds the typed text and the live speech preview.
 *
 * These change many times a second while you speak, and the composer sits
 * inside VoiceBeam and around MetalFx - both of which re-measure and rebuild
 * when their subtree re-renders. Keeping this state in a leaf means the beam
 * and the metal never see those updates at all.
 */
const ComposerField = forwardRef(function ComposerField(
  { recording, busy, ready, onKeyDown, onEmptyChange },
  ref,
) {
  const [text, setText] = useState('')
  const [live, setLive] = useState('')
  const areaRef = useRef(null)

  useImperativeHandle(ref, () => ({
    getText: () => text,
    clear: () => {
      setText('')
      setLive('')
      onEmptyChange(false)
    },
    setLive,
  }), [text, onEmptyChange])

  // Grow with the content, up to the CSS max-height.
  useEffect(() => {
    const el = areaRef.current
    if (!el) return

    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text])

  function onChange(e) {
    const next = e.target.value
    setText(next)
    // Tell the composer only when emptiness actually changes.
    onEmptyChange(next.trim().length > 0)
  }

  return (
    <textarea
      ref={areaRef}
      rows={1}
      maxLength={500}
      value={recording ? live : text}
      onChange={onChange}
      onKeyDown={onKeyDown}
      placeholder={
        recording
          ? 'Listening…'
          : ready
            ? 'Ask me anything..'
            : 'Connecting to the model server…'
      }
      disabled={busy}
      readOnly={recording}
      className={recording ? 'live' : undefined}
    />
  )
})
