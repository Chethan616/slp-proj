import { useEffect, useRef, useState } from 'react'
import { ThinkingOrb } from 'thinking-orbs'
import Composer from './components/Composer'
import Turn from './components/Turn'
import { getInfo, postText, postVoice, wake } from './api'

const SUGGESTIONS = [
  "What's the weather like today?",
  'Set an alarm for 7 am',
  'What is my account balance?',
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

const STAGES = [
  ['record', 'Record'],
  ['stt', 'Speech recognition'],
  ['nlu', 'Intent model'],
  ['reply', 'Response'],
]

export default function App() {
  const [turns, setTurns] = useState([])
  const [phase, setPhase] = useState('idle')
  const [stage, setStage] = useState(null)
  const [error, setError] = useState(null)
  const [info, setInfo] = useState(null)
  const [speak, setSpeak] = useState(false)
  const bottomRef = useRef(null)

  const busy = phase === 'transcribing' || phase === 'classifying'

  // Cloud Run scales to zero, so wake the service while the page is being read
  // rather than making the first question pay the cold start.
  useEffect(() => {
    let cancelled = false
    wake()
      .then(getInfo)
      .then((i) => !cancelled && setInfo(i))
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, phase])

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

  const handleVoice = (blob) => run(() => postVoice(blob), ['transcribing'], 'stt')
  const handleText = (value) => run(() => postText(value), ['classifying'], 'nlu')

  const stageIdx = STAGES.findIndex(([k]) => k === stage)
  const active = PHASES[phase]

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <ThinkingOrb state={busy ? 'working' : 'breathing'} size={20} theme="dark" />
          <div>
            <h1>VoiceBot</h1>
            <p className="tagline">Speech recognition and intent classification</p>
          </div>
        </div>
        <div className="topbar-right">
          <div className="pipeline" aria-label="processing pipeline">
          {STAGES.map(([key, label], i) => (
            <span key={key} style={{ display: 'contents' }}>
              <span
                className={`stage${stage === key ? ' active' : ''}${
                  stageIdx >= 0 && i < stageIdx ? ' done' : ''
                }`}
              >
                {label}
              </span>
              {i < STAGES.length - 1 && <span className="arrow">→</span>}
            </span>
            ))}
          </div>
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
        </div>
      </header>

      <main className={turns.length === 0 ? "empty" : ""}>
        <section className="transcript" aria-live="polite">
          {turns.length === 0 && (
            <div className="hero">
              <div className="hero-orb">
                <ThinkingOrb state={active.orb} size={64} theme="dark" />
              </div>
              <h2>Speak, and it works out what you meant.</h2>
              <p>
                Your speech is transcribed by Whisper, then classified into one of 41 intents by a
                fine-tuned DistilBERT model. Anything outside those intents is detected and refused
                rather than guessed at.
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
        speak={speak}
        setSpeak={setSpeak}
        onVoice={handleVoice}
        onText={handleText}
        onError={setError}
      />
    </div>
  )
}
