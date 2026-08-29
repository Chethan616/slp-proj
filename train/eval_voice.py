"""End-to-end evaluation of the full voice pipeline.

Text-only accuracy does not tell you how the system behaves when the input is
speech, because recognition errors propagate into the classifier. To measure
that without recruiting human speakers, this script synthesises spoken versions
of held-out test utterances with the offline Windows System.Speech voices,
runs the resulting audio through the deployed pipeline, and reports:

  * word error rate of the speech recogniser
  * intent accuracy from audio, versus intent accuracy from the clean text
  * per-utterance latency

Writes results/voice_eval.json and results/voice_eval.png.
"""

import json
import random
import re
import string
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DATA, RESULTS, ROOT, load_labels

sys.path.insert(0, str(ROOT / "app"))
import pipeline  # noqa: E402

AUDIO_DIR = DATA / "eval_audio"
N_PER_INTENT = 2          # utterances sampled per in-scope intent
SEED = 42


def normalise(text: str) -> str:
    """WER is measured on lowercased, unpunctuated text -- Whisper restores
    punctuation and casing that the reference transcripts do not have, and
    counting those as errors would be misleading."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


# Synthesis is driven through Windows' built-in System.Speech engine rather than
# a Python wrapper: the wrapper (pyttsx3) writes its files correctly but its
# event loop does not reliably return control on Windows, which hangs the run.
PS_SYNTH = r"""
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 '{manifest}' | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($synth.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo.Name }})
$synth.Rate = 0
foreach ($item in $items) {{
  $synth.SelectVoice($voices[$item.voice % $voices.Count])
  $synth.SetOutputToWaveFile($item.path)
  $synth.Speak($item.text)
}}
$synth.SetOutputToNull()
$synth.Dispose()
"""


def synthesise(rows) -> list[dict]:
    """Render each utterance to a WAV file with an offline system voice.

    Utterances alternate between the installed voices so the evaluation is not
    tuned to a single speaker.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    n_voices = 2
    plan = [
        {"path": str(AUDIO_DIR / f"{i:03d}.wav"), "text": row["text"], "voice": i % n_voices}
        for i, row in enumerate(rows)
    ]

    todo = [p for p in plan if not Path(p["path"]).exists()]
    if todo:
        manifest = AUDIO_DIR / "_manifest.json"
        manifest.write_text(json.dumps(todo), encoding="utf-8")
        script = PS_SYNTH.format(manifest=str(manifest).replace("'", "''"))
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            raise SystemExit(
                "Speech synthesis failed. This step needs Windows System.Speech "
                f"voices.\n{proc.stderr.strip()[:800]}"
            )
        manifest.unlink(missing_ok=True)

    out = []
    for row, p in zip(rows, plan):
        path = Path(p["path"])
        if not path.exists() or path.stat().st_size < 1000:
            raise SystemExit(f"Synthesis produced no audio for {path.name!r}.")
        out.append({**row, "audio": path, "voice": p["voice"]})
    return out


def main() -> None:
    from jiwer import wer

    random.seed(SEED)
    labels = load_labels()
    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))

    # Stratified sample: a couple of utterances from each in-scope intent, plus
    # a handful of out-of-scope ones.
    by_intent = defaultdict(list)
    for r in rows:
        by_intent[r["label"]].append(r)

    sample = []
    for intent, items in by_intent.items():
        k = N_PER_INTENT if intent != "oos" else 6
        sample.extend(random.sample(items, min(k, len(items))))
    random.shuffle(sample)
    print(f"Evaluating {len(sample)} utterances through the full voice pipeline\n")

    print("Synthesising speech with offline system voices ...")
    sample = synthesise(sample)

    pipeline.warmup()

    refs, hyps, results = [], [], []
    audio_correct = text_correct = 0
    stt_times, infer_times = [], []

    for i, row in enumerate(sample, 1):
        audio_bytes = row["audio"].read_bytes()

        t0 = time.perf_counter()
        stt = pipeline.transcribe(audio_bytes)
        total_ms = (time.perf_counter() - t0) * 1000

        hyp = normalise(stt["transcript"])
        ref = normalise(row["text"])
        refs.append(ref)
        hyps.append(hyp if hyp else "")

        from_audio = pipeline.classify(stt["transcript"]) if hyp else {"intent": "oos", "confidence": 0.0, "infer_ms": 0}
        from_text = pipeline.classify(row["text"])

        a_ok = from_audio["intent"] == row["label"]
        t_ok = from_text["intent"] == row["label"]
        audio_correct += a_ok
        text_correct += t_ok
        stt_times.append(stt["stt_ms"])
        infer_times.append(from_audio["infer_ms"])

        results.append({
            "reference": row["text"],
            "hypothesis": stt["transcript"],
            "true_intent": row["label"],
            "intent_from_audio": from_audio["intent"],
            "intent_from_text": from_text["intent"],
            "audio_correct": bool(a_ok),
            "text_correct": bool(t_ok),
            "utterance_wer": wer(ref, hyp) if ref and hyp else 1.0,
            "stt_ms": stt["stt_ms"],
            "audio_seconds": stt["audio_seconds"],
        })

        flag = " " if a_ok else "x"
        if i % 10 == 0 or not a_ok:
            print(f"  [{flag}] {i:>3}/{len(sample)}  {row['label']:<22} <- {stt['transcript'][:52]!r}")

    corpus_wer = wer(refs, hyps)
    n = len(sample)
    summary = {
        "n_utterances": n,
        "corpus_wer": float(corpus_wer),
        "intent_accuracy_from_audio": audio_correct / n,
        "intent_accuracy_from_clean_text": text_correct / n,
        "accuracy_drop": (text_correct - audio_correct) / n,
        "mean_stt_ms": float(np.mean(stt_times)),
        "median_stt_ms": float(np.median(stt_times)),
        "mean_infer_ms": float(np.mean(infer_times)),
        "mean_audio_seconds": float(np.mean([r["audio_seconds"] for r in results])),
        "real_time_factor": float(np.mean(stt_times) / 1000 / np.mean([r["audio_seconds"] for r in results])),
        "perfect_transcriptions": sum(1 for r in results if r["utterance_wer"] == 0) / n,
    }

    print("\n" + "=" * 62)
    print(f"  utterances evaluated        : {n}")
    print(f"  word error rate             : {corpus_wer:.4f}")
    print(f"  transcribed perfectly       : {summary['perfect_transcriptions']:.1%}")
    print(f"  intent accuracy from audio  : {summary['intent_accuracy_from_audio']:.4f}")
    print(f"  intent accuracy from text   : {summary['intent_accuracy_from_clean_text']:.4f}")
    print(f"  accuracy lost to recognition: {summary['accuracy_drop']:.4f}")
    print(f"  mean STT latency            : {summary['mean_stt_ms']:.0f} ms "
          f"(real-time factor {summary['real_time_factor']:.2f})")
    print("=" * 62)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "voice_eval.json").write_text(
        json.dumps({"summary": summary, "utterances": results}, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(
        ["From clean text", "From synthesised speech"],
        [summary["intent_accuracy_from_clean_text"], summary["intent_accuracy_from_audio"]],
        color=["#6366f1", "#f59e0b"], width=0.55,
    )
    for i, v in enumerate([summary["intent_accuracy_from_clean_text"], summary["intent_accuracy_from_audio"]]):
        axes[0].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Intent accuracy")
    axes[0].set_title(f"Cost of going through speech (n={n})")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].hist([r["utterance_wer"] for r in results], bins=np.arange(0, 1.05, 0.05), color="#22c55e")
    axes[1].set_xlabel("Per-utterance word error rate")
    axes[1].set_ylabel("Utterances")
    axes[1].set_title(f"WER distribution (corpus WER {corpus_wer:.3f})")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "voice_eval.png", dpi=160)
    plt.close(fig)
    print(f"\nWrote results/voice_eval.json and results/voice_eval.png")


if __name__ == "__main__":
    main()
