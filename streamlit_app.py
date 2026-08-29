"""Voice-enabled chatbot — Streamlit deployment.

Same two-stage pipeline as the FastAPI app in app/: Whisper transcribes the
recorded audio, a fine-tuned DistilBERT classifies the transcript into one of 41
intents, and a response is generated for that intent. Both the recognised speech
and the reply are displayed.

Run locally:  streamlit run streamlit_app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

st.set_page_config(
    page_title="Vox — Voice-Enabled Chatbot",
    page_icon="◉",
    layout="centered",
)

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 820px; }
.vox-header { display:flex; align-items:center; gap:14px; margin-bottom:4px; }
.vox-logo { width:44px; height:44px; border-radius:12px; background:#4f46e5; color:#fff;
            display:flex; align-items:center; justify-content:center; font-size:20px; }
.vox-title { margin:0; font-size:1.45rem; font-weight:700; line-height:1.2; }
.vox-sub { margin:0; font-size:.85rem; opacity:.65; }
.vox-pipeline { font-size:.75rem; opacity:.6; margin:14px 0 4px; letter-spacing:.02em; }
.turn { border:1px solid rgba(128,128,128,.25); border-radius:14px;
        padding:14px 16px; margin-bottom:14px; }
.turn .lbl { font-size:.66rem; text-transform:uppercase; letter-spacing:.08em;
             opacity:.55; display:block; margin-bottom:3px; }
.turn .said { font-size:1.02rem; font-weight:600; margin-bottom:10px; }
.turn .reply { font-size:1rem; }
.chip { display:inline-block; padding:2px 10px; border-radius:999px; background:#eef0ff;
        color:#4338ca; font-family:ui-monospace,Menlo,monospace; font-size:.78rem;
        font-weight:600; }
.chip.oos { background:transparent; border:1px dashed #b45309; color:#b45309; }
.meta { font-size:.72rem; opacity:.6; margin-top:9px; padding-top:8px;
        border-top:1px dashed rgba(128,128,128,.28); }
@media (prefers-color-scheme: dark) {
  .chip { background:#23253a; color:#a5b4fc; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="vox-header">
  <div class="vox-logo">◉</div>
  <div>
    <p class="vox-title">Vox — Voice-Enabled Chatbot</p>
    <p class="vox-sub">Speech recognition with Whisper · intent classification with DistilBERT</p>
  </div>
</div>
<p class="vox-pipeline">microphone → Whisper (speech recognition) → DistilBERT (41-way intent) → response</p>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Whisper and DistilBERT (first run takes a moment)…")
def get_pipeline():
    """Load both models once per container, not once per interaction."""
    import pipeline as pl

    pl.warmup()
    return pl


def render_turn(turn: dict) -> None:
    low = turn["intent"] == "oos" or turn.get("below_threshold")
    pct = round(turn["confidence"] * 100)
    alts = " · ".join(
        f"{t['intent']} {round(t['confidence'] * 100)}%" for t in turn.get("top", [])[1:]
    )
    if turn["source"] == "voice":
        stt_meta = f"Whisper · {turn['stt_ms']} ms · {turn['audio_seconds']}s of audio"
    else:
        stt_meta = "typed input"

    st.markdown(
        f"""
<div class="turn">
  <span class="lbl">Recognised speech</span>
  <div class="said">{turn['transcript'] or '(nothing recognised)'}</div>
  <span class="lbl">Intent</span>
  <div><span class="chip {'oos' if low else ''}">{turn['intent']}</span>
       <span style="font-size:.8rem;opacity:.65;"> {pct}% confidence</span></div>
  <div style="margin-top:10px;"><span class="lbl">Vox</span>
       <div class="reply">{turn['reply']}</div></div>
  <div class="meta">{stt_meta} · intent inference {turn['infer_ms']} ms{
      ' · next: ' + alts if alts else ''}</div>
</div>
""",
        unsafe_allow_html=True,
    )


pl = get_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------- input
tab_voice, tab_text = st.tabs(["🎙️  Speak", "⌨️  Type"])

with tab_voice:
    audio = st.audio_input("Record your question", key=f"mic_{len(st.session_state.history)}")
    if audio is not None:
        data = audio.getvalue()
        if len(data) < 1200:
            st.warning("That recording was too short — hold the button a little longer.")
        else:
            with st.spinner("Transcribing and classifying…"):
                t0 = time.perf_counter()
                result = pl.voice_chat(data)
                result["source"] = "voice"
                result["total_ms"] = round((time.perf_counter() - t0) * 1000)
            st.session_state.history.append(result)
            st.rerun()

with tab_text:
    typed = st.chat_input("Type a question instead…")
    if typed:
        result = pl.chat(typed)
        result["transcript"] = typed
        result["source"] = "text"
        result["stt_ms"] = 0
        result["audio_seconds"] = 0
        st.session_state.history.append(result)
        st.rerun()

    st.caption("Try: “what's the weather like today”, “set an alarm for 7 am”, "
               "“what is my account balance”, “tell me a joke”, “who won the world cup in 1994”")

# ---------------------------------------------------------------- output
if st.session_state.history:
    st.markdown("### Conversation")
    for turn in reversed(st.session_state.history):
        render_turn(turn)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
else:
    st.info(
        "Record a question above. Your speech is transcribed on the server by Whisper, "
        "classified into one of 41 intents by a fine-tuned DistilBERT model, and answered. "
        "Questions outside those 41 intents are detected and refused rather than guessed at."
    )

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    info = pl.model_info()
    st.markdown("### System")
    st.markdown(
        f"""
- **Speech recognition:** {info['stt_model']}
- **Intent model:** {info['intent_model']}
- **Intents:** {info['num_intents']} (40 in-scope + out-of-scope)
- **Confidence threshold:** {info['conf_threshold']}
"""
    )
    st.markdown("### Results")
    st.markdown(
        """
| Metric | Value |
|---|---:|
| Intent accuracy | 0.9460 |
| Macro F1 | 0.9556 |
| Word error rate | 0.0394 |
| Accuracy from speech | 0.9767 |
"""
    )
    st.caption(
        "Trained on a 40-intent subset of CLINC150 plus its out-of-scope class. "
        "Full method and results in the project report."
    )
    with st.expander("All 41 intents"):
        st.write(", ".join(info["intents"]))
    st.markdown(
        "[Source code](https://github.com/Chethan616/slp-proj) · "
        "[Model](https://huggingface.co/Chethan616/voice-chatbot-intent-distilbert)"
    )
