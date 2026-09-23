import { useEffect, useRef, useState } from 'react'
import Recorder from './components/Recorder'
import Turn from './components/Turn'
import { getInfo, postText, postVoice, wake } from './api'

const SUGGESTIONS = [
  "What's the weather like today?",
  'Set an alarm for 7 am',
  'What is my account balance?',
  'Tell me a joke',
  'How do I get to the airport?',
  'Who won the world cup in 1994?',
]

const STAGES = ['record', 'stt', 'nlu', 'reply']
const STAGE_LABELS = {
  record: 'Record',
  stt: 'Speech recognition',
  nlu: 'Intent model',
  reply: 'Response',
}

export default function App() {
  const [turns, setTurns] = useState([])
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState(null)
  const [error, setError] = useState(null)
  const [thinking, setThinking] = useState(null)
  const [info, setInfo] = useState(null)
  const [waking, setWaking] = useState(true)
  const [speak, setSpeak] = useState(false)
  const [text, setText] = useState('')
  const bottomRef = useRef(null)

  // Cloud Run scales to zero, so wake the service while the page is being read
  // rather than making the first question pay for the cold start.
  useEffect(() => {
    let cancelled = false
    wake()
      .then(() => getInfo())
      .then((i) => !cancelled && setInfo(i))
      .catch(() => {})
      .finally(() => !cancelled && setWaking(false))
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, thinking])

  function say(reply) {
    if (!speak || !reply || !('speechSynthesis' in window)) return
    const u = new SpeechSynthesisUtterance(reply)
    u.rate = 1.02
    speechSynthesis.cancel()
    speechSynthesis.speak(u)
  }

  async function run(work, stages, thinkingLabel) {
    setError(null)
    setBusy(true)
    setThinking(thinkingLabel)
    setStage(stages[0])
    try {
      const data = await work()
      setStage('reply')
      setTurns((t) => [...t, data])
      say(data.reply)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      setThinking(null)
      setTimeout(() => setStage(null), 900)
    }
  }

  const handleRecorded = (blob) =>
    run(() => postVoice(blob), ['stt'], 'Transcribing and classifying')

  function handleSubmit(e) {
    e.preventDefault()
    const value = text.trim()
    if (!value || busy) return
    setText('')
    run(() => postText(value), ['nlu'], 'Classifying')
  }

  const stageIdx = STAGES.indexOf(stage)

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">◉</span>
          <div>
            <h1>VoiceBot</h1>
            <p className="tagline">Voice-enabled chatbot · Whisper + DistilBERT</p>
          </div>
        </div>
        <div className="pipeline" aria-label="processing pipeline">
          {STAGES.map((s, i) => (
            <span key={s}>
              <span
                className={`stage${stage === s ? ' active' : ''}${
                  stageIdx >= 0 && i < stageIdx ? ' done' : ''
                }`}
              >
                {STAGE_LABELS[s]}
              </span>
              {i < STAGES.length - 1 && <span className="arrow"> → </span>}
            </span>
          ))}
        </div>
      </header>

      <main>
        <section className="transcript" aria-live="polite">
          {turns.length === 0 && !thinking && (
            <div className="empty-state">
              <p className="empty-title">Press the microphone and speak.</p>
              <p className="empty-sub">
                Your speech is transcribed on the server by Whisper, classified into one of 41
                intents by a fine-tuned DistilBERT model, and answered. Questions outside those
                intents are detected and refused rather than guessed at.
              </p>
              <ul className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <li key={s}>
                    <button
                      onClick={() => run(() => postText(s), ['nlu'], 'Classifying')}
                      disabled={busy}
                    >
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {waking && turns.length === 0 && (
            <div className="waking">Waking the model server — this takes a moment on first load.</div>
          )}

          {turns.map((t, i) => (
            <Turn key={i} data={t} />
          ))}

          {thinking && <div className="thinking">{thinking}</div>}
          {error && <div className="error">{error}</div>}
          <div ref={bottomRef} />
        </section>
      </main>

      <footer className="controls">
        <Recorder disabled={busy} onRecorded={handleRecorded} onError={setError} />

        <form className="text-row" onSubmit={handleSubmit} autoComplete="off">
          <input
            type="text"
            maxLength={500}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="…or type here if you have no microphone"
          />
          <button type="submit" className="send" disabled={busy}>
            Send
          </button>
        </form>

        <div className="footer-meta">
          <label className="speak-toggle">
            <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
            Read replies aloud
          </label>
          <span className="model-info">
            {info
              ? `${info.stt_model} → ${info.intent_model} · ${info.num_intents} intents`
              : 'connecting…'}
          </span>
          <span className="model-info">
            <a href="https://github.com/Chethan616/slp-proj" target="_blank" rel="noreferrer">
              source
            </a>
          </span>
        </div>
      </footer>
    </div>
  )
}
