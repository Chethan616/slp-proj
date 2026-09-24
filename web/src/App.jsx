import { useEffect, useRef, useState } from 'react'
import { MetalFx, MetalText, useMetalBend } from 'metal-fx'
import { ThinkingOrb } from 'thinking-orbs'
import Composer from './components/Composer'
import Turn from './components/Turn'
import { getInfo, postText, postVoice, wake } from './api'

/* One per domain, so a few clicks walk through the whole label space and show
 * a different avatar each time. The last is deliberately unanswerable: it
 * demonstrates the out-of-scope class rather than hiding it. */
const SUGGESTIONS = [
  "What's the weather like today?",
  'Set an alarm for 7 am',
  'Book me a flight to London',
  'How long will it take me to drive there?',
  'Add milk to my shopping list',
  'Find me a good place for dinner',
  'When is my next payday?',
  'Tell me a joke',
  'Who won the world cup in 1994?',
]

/* Each pipeline phase gets the orb state that actually describes it, so the
 * animation reports what the system is doing rather than decorating it. */
const PHASES = {
  idle: { orb: 'breathing', label: null },
  listening: { orb: 'listening', label: 'Listening' },
  transcribing: { orb: 'working', label: 'Transcribing speech' },
  classifying: { orb: 'solving', label: 'Classifying intent' },
}

/* The strip names each stage and the technique that performs it, and stays on
 * screen for the whole session. The hero paragraph explains the same thing in
 * prose, but it is replaced by the conversation on the first turn, so without
 * this the pipeline would be invisible during an actual demonstration. */
const STAGES = [
  { key: 'record', label: 'Voice input', short: 'Voice', tech: 'microphone' },
  { key: 'stt', label: 'Speech recognition', short: 'Whisper', tech: 'Whisper base.en' },
  { key: 'nlu', label: 'Intent classification', short: 'DistilBERT', tech: 'DistilBERT, 41 intents' },
  { key: 'reply', label: 'Response', short: 'Reply', tech: 'generated reply' },
]

function Pipeline({ stage }) {
  const idx = STAGES.findIndex((s) => s.key === stage)

  return (
    <div className="pipeline" aria-label="Processing pipeline">
      {STAGES.map((s, i) => (
        <div className="pipe-step-wrap" key={s.key}>
          <div
            className={`pipe-step${stage === s.key ? ' active' : ''}${
              idx >= 0 && i < idx ? ' done' : ''
            }`}
          >
            <span className="pipe-label">{s.label}</span>
            <span className="pipe-short">{s.short}</span>
            <span className="pipe-tech">{s.tech}</span>
          </div>
          {i < STAGES.length - 1 && <span className="pipe-arrow" aria-hidden="true">→</span>}
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const [turns, setTurns] = useState([])
  const [phase, setPhase] = useState('idle')
  const [stage, setStage] = useState(null)
  const [error, setError] = useState(null)
  const [info, setInfo] = useState(null)
  // 'connecting' until the backend answers, then 'ready'; 'failed' once the
  // retries are exhausted. Every control is inert until this is 'ready'.
  const [conn, setConn] = useState('connecting')
  const [speak, setSpeak] = useState(false)
  const bottomRef = useRef(null)
  const ghRef = useRef(null)

  // Same cursor-driven liquid dent as the send button.
  useMetalBend(ghRef)

  const busy = phase === 'transcribing' || phase === 'classifying'

  // Cloud Run scales to zero, so wake the service while the page is being read
  // rather than making the first question pay the cold start.
  useEffect(() => {
    let cancelled = false
    let attempt = 0

    /* The backend scales to zero, so the first request after an idle period
     * has to start a container and load both models. That takes longer than a
     * single request will wait, hence the retries with a growing pause. */
    async function connect() {
      while (!cancelled && attempt < 6) {
        attempt += 1
        try {
          await wake()
          const i = await getInfo()
          if (!cancelled) {
            setInfo(i)
            setConn('ready')
          }
          return
        } catch {
          await new Promise((r) => setTimeout(r, Math.min(2000 * attempt, 8000)))
        }
      }
      if (!cancelled) setConn('failed')
    }

    connect()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, phase])

  const ready = conn === 'ready'

  /* Wraps any action so that pressing a control before the backend is up
   * explains itself rather than failing silently or throwing. */
  function guarded(run) {
    return (...args) => {
      if (!ready) {
        setError(
          conn === 'failed'
            ? 'The model server is not responding. Give it a moment and reload the page.'
            : 'Still connecting to the model server - one moment.',
        )
        return
      }
      run(...args)
    }
  }

  function say(reply) {
    if (!speak || !reply || !('speechSynthesis' in window)) return
    const u = new SpeechSynthesisUtterance(reply)
    u.rate = 1.02
    speechSynthesis.cancel()
    speechSynthesis.speak(u)
  }

  async function run(work, phases, firstStage) {
    setError(null)
    setPhase(phases[0])
    setStage(firstStage)
    try {
      const data = await work()
      setStage('reply')
      setTurns((t) => [...t, data])
      say(data.reply)
    } catch (err) {
      setError(err.message)
    } finally {
      setPhase('idle')
      setTimeout(() => setStage(null), 900)
    }
  }

  const handleVoice = guarded((blob) => run(() => postVoice(blob), ['transcribing'], 'stt'))
  const handleText = guarded((value) => run(() => postText(value), ['classifying'], 'nlu'))

  const active = PHASES[phase]

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <ThinkingOrb state={busy ? 'working' : 'breathing'} size={20} theme="dark" />
          <div>
            <h1>
              <MetalText
                className="metal-text"
                font="600 16px/1.2 Inter, system-ui, sans-serif"
                color="#ffffff"
                theme="dark"
              >
                VoiceBot
              </MetalText>
            </h1>
            <p className="tagline">Whisper → DistilBERT</p>
          </div>
        </div>
        <div className="topbar-right">
          <MetalFx
            ref={ghRef}
            preset="chromatic"
            variant="circle"
            theme="dark"
            innerShadow
            strength={0.85}
          >
          <a
            className="gh-link"
            href="https://github.com/Chethan616/slp-proj"
            target="_blank"
            rel="noreferrer"
            aria-label="Source code on GitHub"
            title="Source on GitHub"
          >
            <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
              <path
                fill="currentColor"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                   0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                   -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                   .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                   -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
                   1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82
                   1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                   1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
              />
            </svg>
          </a>
          </MetalFx>
        </div>
      </header>

      <Pipeline stage={stage} />

      <main className={turns.length === 0 ? "empty" : ""}>
        <section className="transcript" aria-live="polite">
          {turns.length === 0 && (
            <div className="hero">
              <div className="hero-orb">
                <ThinkingOrb state={active.orb} size={64} theme="dark" />
              </div>
              <h2>
                Speak, and it works out <em>what you meant</em>.
              </h2>
              <p>
                Press the microphone and talk. Your speech is transcribed by Whisper, then
                classified into one of 41 intents by a fine-tuned DistilBERT model. Anything
                outside those intents is detected and refused rather than guessed at.
              </p>
              <ul className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <li key={s}>
                    <button onClick={() => handleText(s)} disabled={busy}>
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {turns.map((t, i) => (
            <Turn key={i} data={t} />
          ))}

          {active.label && turns.length > 0 && (
            <div className="status-row">
              <ThinkingOrb state={active.orb} size={20} theme="dark" />
              <span>{active.label}…</span>
            </div>
          )}

          {error && <div className="error">{error}</div>}
          <div ref={bottomRef} />
        </section>
      </main>

      <Composer
        busy={busy}
        processing={busy}
        info={info}
        ready={ready}
        conn={conn}
        onNotReady={() =>
          setError(
            conn === 'failed'
              ? 'The model server is not responding. Give it a moment and reload the page.'
              : 'Still connecting to the model server - one moment.',
          )
        }
        speak={speak}
        setSpeak={setSpeak}
        onVoice={handleVoice}
        onText={handleText}
        onError={setError}
      />
    </div>
  )
}
