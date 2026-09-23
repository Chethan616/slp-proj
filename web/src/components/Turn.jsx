import { BotAvatar } from 'bot-avatars'

/* One exchange. The assignment requires the recognised speech and the reply to
 * both be shown; the intent, its confidence and the runners-up are shown too so
 * the model's decision is visible rather than implied. */
export default function Turn({ data }) {
  const pct = Math.round(data.confidence * 100)
  const low = data.below_threshold || data.intent === 'oos'
  const alts = (data.top || [])
    .slice(1)
    .map((t) => `${t.intent} ${Math.round(t.confidence * 100)}%`)
    .join('  ')

  return (
    <div className="turn">
      <div className="said">
        <div className="text">{data.transcript || '(nothing recognised)'}</div>
      </div>

      <div className="reply-row">
        <div className="reply-avatar">
          <BotAvatar type="clover" size={34} state="default" seed={0.2} />
        </div>

        <div className="reply-body">
          <div className="text">{data.reply}</div>

          <div className="readout">
            {data.top?.length > 0 && (
              <>
                <span className={`chip${low ? ' oos' : ''}`}>{data.intent}</span>
                <span className={`bar${low ? ' low' : ''}`}>
                  <i style={{ width: `${pct}%` }} />
                </span>
                <span>{pct}%</span>
                <span className="sep">·</span>
              </>
            )}
            <span>
              {data.source === 'voice'
                ? `heard in ${data.stt_ms} ms from ${data.audio_seconds}s of audio`
                : 'typed'}
            </span>
            <span className="sep">·</span>
            <span>intent {data.infer_ms} ms</span>
            {alts && (
              <>
                <span className="sep">·</span>
                <span>next: {alts}</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
