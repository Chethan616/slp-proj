/* One exchange: what the recogniser heard, what the classifier decided, and the
 * reply. The assignment asks for the recognised speech and the response to both
 * be displayed; the intent and confidence are shown too so the model's decision
 * is visible rather than implied. */
export default function Turn({ data }) {
  const pct = Math.round(data.confidence * 100)
  const low = data.below_threshold || data.intent === 'oos'
  const alts = (data.top || [])
    .slice(1)
    .map((t) => `${t.intent} ${Math.round(t.confidence * 100)}%`)
    .join(' · ')

  const sttMeta =
    data.source === 'voice'
      ? `Whisper · ${data.stt_ms} ms · ${data.audio_seconds}s of audio`
      : 'typed input'

  return (
    <div className="turn">
      <div className="bubble user">
        <span className="label">Recognised speech</span>
        <div className="text">{data.transcript || '(nothing recognised)'}</div>
        <div className="meta">
          <span>{sttMeta}</span>
        </div>
      </div>

      {data.top?.length > 0 && (
        <div className="intent-row">
          <span>Intent</span>
          <span className={`chip${low ? ' oos' : ''}`}>{data.intent}</span>
          <span className={`bar${low ? ' low' : ''}`}>
            <i style={{ width: `${pct}%` }} />
          </span>
          <span>{pct}%</span>
          {alts && <span className="alt">next: {alts}</span>}
          <span>· {data.infer_ms} ms</span>
        </div>
      )}

      <div className="bubble bot">
        <span className="label">VoiceBot</span>
        <div className="text">{data.reply}</div>
      </div>
    </div>
  )
}
