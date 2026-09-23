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
