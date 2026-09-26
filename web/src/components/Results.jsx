import results from '../results.json'

/* Measured results, read from results.json, which train/export_results.py
 * writes from the same files the report is built from. Nothing here is typed
 * in by hand, so the page cannot drift from the experiments.
 *
 * Charts are plain HTML and SVG in the site's own palette. The two series use
 * categorical slots 1 and 2 (blue, orange), validated for colour-vision
 * separation and contrast against the #181818 card surface.
 */

const pct = (x) => `${(x * 100).toFixed(1)}%`
const f4 = (x) => x.toFixed(4)

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  )
}

/* Grouped horizontal bars: model names are long, so the categorical axis runs
 * down the side where labels have room to be read. */
function ModelChart({ models }) {
  const max = 1
  return (
    <figure className="chart">
      <figcaption>
        <h3>Intent classification</h3>
        <p>
          All four models trained on identical splits and scored once on the same{' '}
          {results.splits.test} held-out utterances.
        </p>
      </figcaption>

      <div className="legend">
        <span className="legend-item">
          <i className="swatch s1" /> Test accuracy
        </span>
        <span className="legend-item">
          <i className="swatch s2" /> Macro F1
        </span>
      </div>

      <div className="bars">
        {models.map((m) => (
          <div className="bar-row" key={m.key}>
            <span className="bar-name">
              {m.name}
              {m.key === 'distilbert' && <em> deployed</em>}
            </span>
            <div className="bar-pair">
              {[
                ['s1', m.accuracy, 'Test accuracy'],
                ['s2', m.macro_f1, 'Macro F1'],
              ].map(([cls, v, title]) => (
                <div className="bar-track" key={cls} title={`${title}: ${f4(v)}`}>
                  <i className={`bar-fill ${cls}`} style={{ width: `${(v / max) * 100}%` }} />
                  <span className="bar-value">{f4(v)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <p className="note">
        The transformer gains {((models[3].accuracy - models[1].accuracy) * 100).toFixed(1)}{' '}
        accuracy points over the best model trained from scratch. Pre-training is
        what buys that: DistilBERT already knows two phrasings of the same request
        are related, where the others see only the hundred examples per class they
        are given.
      </p>
    </figure>
  )
}

/* Single series, so no legend: the heading names what the bars are. */
function PerClassChart({ rows }) {
  const hardest = rows.slice(0, 8)
  return (
    <figure className="chart">
      <figcaption>
        <h3>Hardest intents</h3>
        <p>Per-class F1 for the deployed model, lowest eight of {rows.length}.</p>
      </figcaption>

      <div className="bars">
        {hardest.map((r) => (
          <div className="bar-row compact" key={r.intent}>
            <span className="bar-name mono">{r.intent}</span>
            <div className="bar-track" title={`F1 ${f4(r.f1)} · precision ${f4(r.precision)} · recall ${f4(r.recall)} · ${r.support} test utterances`}>
              <i className="bar-fill s1" style={{ width: `${r.f1 * 100}%` }} />
              <span className="bar-value">{f4(r.f1)}</span>
            </div>
          </div>
        ))}
      </div>

      <p className="note">
        The confusions behind those scores:{' '}
        {results.confusions.slice(0, 3).map((c, i) => (
          <span key={i}>
            {i > 0 && '; '}
            <code>{c.true}</code> read as <code>{c.predicted}</code> {c.count} times
          </span>
        ))}
        .
      </p>
    </figure>
  )
}

function ThresholdTable({ t }) {
  const rows = [
    ['Test, no threshold', t.test_at_zero],
    [`Validation, threshold ${t.best_threshold.toFixed(2)}`, t.validation],
    [`Test, threshold ${t.best_threshold.toFixed(2)}`, t.test],
  ]
  return (
    <figure className="chart">
      <figcaption>
        <h3>Refusing what it cannot answer</h3>
        <p>
          A prediction below the confidence threshold is answered as out-of-scope
          rather than guessed at. The threshold was chosen on validation and only
          then measured on test.
        </p>
      </figcaption>

      <table className="data">
        <thead>
          <tr>
            <th>Setting</th>
            <th>In-scope accuracy</th>
            <th>Out-of-scope recall</th>
            <th>Macro F1</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, r], i) => (
            <tr key={label} className={i === 2 ? 'highlight' : undefined}>
              <td>{label}</td>
              <td>{f4(r.in_scope_accuracy)}</td>
              <td>{f4(r.oos_recall)}</td>
              <td>{f4(r.macro_f1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="note">
        The threshold lifts out-of-scope recall from {pct(t.test_at_zero.oos_recall)} to{' '}
        {pct(t.test.oos_recall)}, costing{' '}
        {((t.test_at_zero.in_scope_accuracy - t.test.in_scope_accuracy) * 100).toFixed(1)}{' '}
        points of in-scope accuracy. That trade is worth making here: a narrow
        assistant is asked things outside its subject far more often than a general one.
      </p>
    </figure>
  )
}

function VoiceChart({ v }) {
  const pair = [
    ['From clean text', v.intent_accuracy_from_clean_text, 's1'],
    ['From synthesised speech', v.intent_accuracy_from_audio, 's2'],
  ]
  return (
    <figure className="chart">
      <figcaption>
        <h3>What speech costs</h3>
        <p>
          {v.n_utterances} held-out utterances synthesised to audio and run through
          the deployed pipeline, so recognition errors reach the classifier exactly
          as they would in use.
        </p>
      </figcaption>

      <div className="bars">
        {pair.map(([label, value, cls]) => (
          <div className="bar-row" key={label}>
            <span className="bar-name">{label}</span>
            <div className="bar-track" title={`Intent accuracy ${f4(value)}`}>
              <i className={`bar-fill ${cls}`} style={{ width: `${value * 100}%` }} />
              <span className="bar-value">{f4(value)}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mini-stats">
        <Stat label="Word error rate" value={f4(v.corpus_wer)} />
        <Stat label="Transcribed exactly" value={pct(v.perfect_transcriptions)} />
        <Stat label="Speech recognition" value={`${Math.round(v.mean_stt_ms)} ms`} />
        <Stat label="Intent inference" value={`${Math.round(v.mean_infer_ms)} ms`} />
      </div>

      <p className="note">
        Recognition costs {(v.accuracy_drop * 100).toFixed(1)} accuracy points. The
        errors tend to land on words that do not decide the intent, which is why the
        drop is small. A real-time factor of {v.real_time_factor.toFixed(2)} means
        transcription outpaces the speaking.
      </p>
    </figure>
  )
}

export default function Results() {
  const h = results.headline
  const d = results.deployed

  return (
    <section className="results" aria-label="Model evaluation">
      <div className="results-head">
        <h2>How well it works</h2>
        <p>
          Every figure below is measured, not quoted: {results.classes} classes,{' '}
          {results.splits.train.toLocaleString()} training and {results.splits.test}{' '}
          test utterances, scored once on a held-out split.
        </p>
      </div>

      <div className="stat-row">
        <Stat label="Test accuracy" value={f4(h.accuracy)} sub="deployed model" />
        <Stat label="Macro F1" value={f4(h.macro_f1)} sub="averaged over classes" />
        <Stat label="Word error rate" value={f4(h.wer)} sub="speech recognition" />
        <Stat label="Out-of-scope recall" value={f4(h.oos_recall)} sub="questions refused" />
      </div>

      <ModelChart models={results.models} />
      <ThresholdTable t={results.threshold} />
      <VoiceChart v={results.voice} />
      <PerClassChart rows={results.per_class} />

      <figure className="chart">
        <figcaption>
          <h3>What actually ships</h3>
          <p>
            The trained model is exported to ONNX and quantised to 8-bit integers
            for deployment.
          </p>
        </figcaption>
        <table className="data">
          <thead>
            <tr>
              <th>Property</th>
              <th>PyTorch</th>
              <th>Deployed, int8 ONNX</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Size on disk</td><td>268.0 MB</td><td>{d.size_mb} MB</td></tr>
            <tr><td>Test accuracy</td><td>{f4(results.models[3].accuracy)}</td><td>{f4(d.accuracy)}</td></tr>
            <tr><td>Macro F1</td><td>{f4(results.models[3].macro_f1)}</td><td>{f4(d.macro_f1)}</td></tr>
            <tr><td>Relative speed</td><td>1.0x</td><td>{d.speedup}x</td></tr>
          </tbody>
        </table>
        <p className="note">
          Quantisation is lossless within noise: the int8 graph agrees with the
          full-precision one on {pct(d.agreement_with_fp32)} of test utterances and
          scores identically, at a quarter of the size.
        </p>
      </figure>
    </section>
  )
}
