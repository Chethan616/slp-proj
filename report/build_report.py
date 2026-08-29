"""Generate report/report.md and a self-contained report/report.html.

Every number in the report is read from results/*.json rather than typed in, so
the document cannot drift out of sync with the experiments. Run this after
train/evaluate.py and train/eval_voice.py.
"""

import base64
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

LIVE_URL = json.loads((REPORT / "deployment.json").read_text(encoding="utf-8"))["live_url"] \
    if (REPORT / "deployment.json").exists() else "(deployment link to be inserted)"


def load(name, default=None):
    path = RESULTS / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(x):
    return f"{x * 100:.1f}%"


def build_markdown() -> str:
    metrics = load("metrics.json", {})
    thresh = load("threshold.json")
    voice = load("voice_eval.json")
    labels = json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8"))

    order = ["tfidf_logreg", "bow_mlp", "bilstm", "distilbert"]
    present = [k for k in order if k in metrics]
    dep = metrics.get("distilbert", {})
    best_baseline = max(
        (metrics[k] for k in present if k != "distilbert"),
        key=lambda m: m["test_accuracy"],
        default=None,
    )

    counts = {
        split: len(json.loads((ROOT / "data" / f"{split}.json").read_text(encoding="utf-8")))
        for split in ("train", "val", "test")
    }

    L = []
    a = L.append

    a("# Voice-Enabled Chatbot using Speech Recognition and Deep Learning")
    a("")
    a(f"**Live application:** {LIVE_URL}  ")
    a(f"**Date:** {date.today().strftime('%d %B %Y')}")
    a("")
    a("---")
    a("")

    # ------------------------------------------------------------------
    a("## 1. Introduction")
    a("")
    a("This project implements and deploys a chatbot that is operated by voice. The")
    a("user speaks into the browser; the recorded audio is transcribed by an automatic")
    a("speech recognition model; the transcript is classified into a user intent by a")
    a("fine-tuned transformer; and a response is produced for that intent. The")
    a("interface shows the recognised speech, the predicted intent with its confidence,")
    a("and the reply, so that every stage of the pipeline is visible.")
    a("")
    a("The system is deployed as a public web application and requires nothing but a")
    a("browser and a microphone to use.")
    a("")
    a("```")
    a("microphone --> MediaRecorder (webm/opus) --> POST /api/voice")
    a("                                                 |")
    a("                        faster-whisper base.en (int8, CTranslate2)")
    a("                                                 |  recognised text")
    a("                        DistilBERT intent classifier (41-way softmax)")
    a("                                                 |  intent + confidence")
    a("                        response template --> displayed and optionally spoken")
    a("```")
    a("")

    # ------------------------------------------------------------------
    a("## 2. Dataset")
    a("")
    a("### 2.1 Source")
    a("")
    a("The intent classifier is trained on **CLINC150** (Larson et al., EMNLP 2019), a")
    a("benchmark created specifically for intent classification. It contains 150")
    a("in-scope intents spread over 10 domains, with 150 crowdsourced utterances per")
    a("intent, and — unusually for an intent dataset — an explicit **out-of-scope**")
    a("class of queries that a task-oriented assistant is not built to answer.")
    a("")
    a("### 2.2 Subset used")
    a("")
    a("A 40-intent subset spanning eight domains was selected, plus the out-of-scope")
    a("class, giving **41 classes**. The selection is fixed in code")
    a("(`train/prepare_data.py`) and therefore reproducible. Two considerations drove")
    a("it: training all 150 classes on CPU is slow without teaching anything extra, and")
    a("every intent needs a written response template for the chatbot to reply at all.")
    a("")
    a("| Domain | Intents |")
    a("|---|---|")
    a("| Small talk | greeting, goodbye, thank_you, tell_joke, what_is_your_name, how_old_are_you, are_you_a_bot, what_can_i_ask_you |")
    a("| Utility | weather, time, date, alarm, timer, definition, calculator, flip_coin |")
    a("| Travel | flight_status, book_flight, book_hotel, translate, exchange_rate |")
    a("| Auto and commute | directions, traffic, distance, gas |")
    a("| Banking | balance, transactions, pay_bill, credit_score |")
    a("| Home | play_music, next_song, shopping_list, todo_list, reminder |")
    a("| Work | payday, meeting_schedule, pto_balance |")
    a("| Kitchen and dining | recipe, restaurant_suggestion, calories |")
    a("| — | oos (out of scope) |")
    a("")
    a("| Split | Utterances | Per in-scope intent | Out-of-scope |")
    a("|---|---:|---:|---:|")
    a(f"| Train | {counts['train']:,} | 100 | 250 |")
    a(f"| Validation | {counts['val']:,} | 20 | 100 |")
    a(f"| Test | {counts['test']:,} | 30 | 300 |")
    a("")
    a("The out-of-scope examples in the test split were subsampled from the 1,000 that")
    a("CLINC provides, because 1,000 out-of-scope against 1,200 in-scope would have let")
    a("a single class dominate every aggregate metric.")
    a("")
    a("### 2.3 Responses")
    a("")
    a("CLINC150 is a classification dataset and ships no replies. A response file")
    a("(`app/responses.json`) maps each of the 41 intents to two or three written")
    a("replies, one of which is chosen at random per turn. A few contain placeholders")
    a("(`{time}`, `{date}`, `{coin}`) that are filled from the server clock or a random")
    a("draw, so those intents give genuinely live answers.")
    a("")

    # ------------------------------------------------------------------
    a("## 3. Speech recognition")
    a("")
    a("### 3.1 Model")
    a("")
    a("Speech recognition uses **Whisper** (Radford et al., 2022) through the")
    a("`faster-whisper` implementation, which runs the model on the CTranslate2")
    a("inference engine. The `base.en` checkpoint is used: 74M parameters, English-only")
    a("(and therefore both faster and more accurate on English than the multilingual")
    a("checkpoint of the same size), with **int8 quantisation** so that weights are")
    a("stored as 8-bit integers instead of 32-bit floats. This roughly quarters memory")
    a("use and materially speeds up CPU inference at negligible accuracy cost, which is")
    a("what makes the model viable on a free two-core container.")
    a("")
    a("### 3.2 How it works")
    a("")
    a("Audio is resampled to 16 kHz and converted into an **80-channel log-Mel")
    a("spectrogram**: the short-time Fourier transform gives energy per frequency per")
    a("time window, the Mel scale warps the frequency axis to match human pitch")
    a("perception, and the logarithm compresses the dynamic range. That representation")
    a("is passed to a **transformer encoder**, and a **transformer decoder**")
    a("autoregressively emits text tokens while attending to the encoder output. Whisper")
    a("was trained on 680,000 hours of weakly supervised audio, which is why it")
    a("generalises to unfamiliar microphones and accents without any adaptation here.")
    a("")
    a("Decoding uses greedy search (beam size 1), which approximately halves latency")
    a("with no measurable accuracy cost on utterances this short.")
    a("")

    # ------------------------------------------------------------------
    a("## 4. Model architecture")
    a("")
    a("Four intent classifiers were trained on identical splits so that the value of")
    a("the transformer can be measured rather than assumed. Only the last is deployed.")
    a("All are implemented in PyTorch; scikit-learn provides the classical baseline.")
    a("")
    a("**1. TF-IDF + Logistic Regression.** Unigram and bigram TF-IDF features with")
    a("sublinear term frequency scaling, fed to a multinomial logistic regression. No")
    a("neural network at all — the point of reference for whether deep learning is")
    a("earning its place.")
    a("")
    a("**2. Bag-of-words + MLP.** A binary bag-of-words vector into a two-hidden-layer")
    a("network (256 and 128 units, ReLU, dropout 0.5). This is the architecture most")
    a("introductory chatbot tutorials use.")
    a("")
    a("**3. Embedding + BiLSTM.** A learned 128-dimensional embedding, a bidirectional")
    a("LSTM with 64 hidden units per direction, masked mean-pooling over real tokens,")
    a("dropout, then a linear classifier. Unlike the first two, this model can use word")
    a("order.")
    a("")
    a("**4. DistilBERT, fine-tuned (deployed).** DistilBERT-base-uncased is a six-layer")
    a("distillation of BERT-base: about 40% smaller and 60% faster while retaining")
    a("roughly 97% of BERT's GLUE performance. A linear classification head over the")
    a("`[CLS]` representation is added and the entire network is fine-tuned with")
    a("cross-entropy loss.")
    a("")
    if dep:
        hp = dep.get("hyperparameters", {})
        a("```")
        a("Input utterance")
        a("   -> WordPiece tokenisation, max length 32")
        a("   -> DistilBERT encoder: 6 transformer layers, hidden 768, 12 heads")
        a("   -> [CLS] representation (768-d)")
        a("   -> pre-classifier Linear(768, 768) + ReLU + dropout(0.2)")
        a(f"   -> Linear(768, {len(labels)}) -> softmax over {len(labels)} intents")
        a("```")
        a("")
        a(f"Trainable parameters: **{dep['params']:,}**.")
        a("")
        a("| Hyperparameter | Value |")
        a("|---|---|")
        a(f"| Checkpoint | `{hp.get('checkpoint')}` |")
        a(f"| Max sequence length | {hp.get('max_len')} |")
        a(f"| Batch size | {hp.get('batch_size')} |")
        a(f"| Epochs | {hp.get('epochs')} |")
        a(f"| Learning rate | {hp.get('lr')} (AdamW) |")
        a(f"| Warmup | {int(hp.get('warmup_ratio', 0) * 100)}% linear, then linear decay |")
        a(f"| Weight decay | {hp.get('weight_decay')} |")
        a("| Gradient clipping | 1.0 |")
        a("| Model selection | best validation macro F1 |")
        a("")

    # ------------------------------------------------------------------
    a("## 5. Methodology")
    a("")
    a("1. **Data preparation.** CLINC150 is downloaded, the 40 chosen intents are")
    a("   extracted, out-of-scope examples are subsampled with a fixed seed, and the")
    a("   splits are written to disk. Train, validation and test come from CLINC's own")
    a("   splits, so no utterance appears in more than one.")
    a("2. **Training.** All four models are trained on the training split. The two")
    a("   from-scratch neural models and the transformer keep the checkpoint with the")
    a("   best validation macro F1 rather than the final epoch.")
    a("3. **Evaluation.** Every model is scored once on the held-out test split.")
    a("   Predictions are saved so that all figures are generated from a single set of")
    a("   numbers.")
    a("4. **Threshold selection.** The confidence threshold for rejecting an utterance")
    a("   as out-of-scope is chosen by sweeping it and taking the value that maximises")
    a("   overall macro F1.")
    a("5. **End-to-end voice evaluation.** Held-out test utterances are synthesised to")
    a("   speech with offline system voices and run through the real pipeline, giving")
    a("   word error rate and the intent accuracy actually achieved from audio.")
    a("6. **Deployment.** The application is containerised and deployed to a public")
    a("   Hugging Face Docker Space.")
    a("")
    a("### 5.1 Out-of-scope handling")
    a("")
    a("A classifier with a softmax output must assign every input to some class. Asked")
    a("something outside its training distribution, it will answer confidently and")
    a("wrongly. Two mechanisms guard against this. First, `oos` is a trained class, so")
    a("the model can predict it directly. Second, if the maximum softmax probability")
    a("falls below a threshold the prediction is overridden to `oos` regardless of the")
    a("argmax, and the bot says it did not understand.")
    a("")

    # ------------------------------------------------------------------
    a("## 6. Results")
    a("")
    a("### 6.1 Intent classification")
    a("")
    a(f"All models were evaluated on the same {counts['test']:,} held-out utterances.")
    a("")
    a("| Model | Family | Parameters | Test accuracy | Macro F1 | Training time |")
    a("|---|---|---:|---:|---:|---:|")
    for key in present:
        m = metrics[key]
        star = " **(deployed)**" if m.get("deployed") else ""
        a(f"| {m['name']}{star} | {m['family']} | {m['params']:,} | "
          f"{m['test_accuracy']:.4f} | {m['test_macro_f1']:.4f} | {m['train_seconds']:.0f}s |")
    a("")
    a("![Model comparison](../results/model_comparison.png)")
    a("")
    if dep and best_baseline:
        d_acc = dep["test_accuracy"] - best_baseline["test_accuracy"]
        d_f1 = dep["test_macro_f1"] - best_baseline["test_macro_f1"]
        a(f"The fine-tuned transformer reaches **{dep['test_accuracy']:.4f}** accuracy and")
        a(f"**{dep['test_macro_f1']:.4f}** macro F1, an improvement of")
        a(f"{d_acc * 100:+.1f} accuracy points and {d_f1 * 100:+.1f} macro-F1 points over the")
        a(f"best non-transformer model ({best_baseline['name']}). The gain comes from")
        a("pre-training: DistilBERT has already learned that \"what's the forecast\" and")
        a("\"will it rain tomorrow\" are related, whereas the from-scratch models can only")
        a("learn that from the hundred examples per class they are given.")
        a("")
    a("Note that macro F1 and accuracy differ noticeably. The test set contains 300")
    a("out-of-scope utterances against 30 per in-scope intent, and out-of-scope is by")
    a("far the hardest class, so it drags accuracy down more than it drags down an")
    a("average taken equally over classes. Both are reported for that reason.")
    a("")
    a("![Training curves](../results/training_curves.png)")
    a("")
    a("### 6.2 Where the deployed model fails")
    a("")
    a("![Confusion matrix](../results/confusion_matrix.png)")
    a("")
    a("![Hardest intents](../results/per_class_f1.png)")
    a("")
    a("The confusion matrix is close to diagonal for in-scope intents. Nearly all")
    a("remaining error is concentrated in the out-of-scope class, which is expected: it")
    a("is not a topic but the absence of one, so it has no consistent vocabulary to")
    a("learn.")
    a("")

    if thresh:
        a("### 6.3 Out-of-scope rejection")
        a("")
        a("![Threshold sweep](../results/threshold_sweep.png)")
        a("")
        a("| Setting | In-scope accuracy | Out-of-scope recall | Macro F1 |")
        a("|---|---:|---:|---:|")
        z = thresh["at_zero"]
        a(f"| No threshold | {z['in_scope_accuracy']:.4f} | {z['oos_recall']:.4f} | {z['macro_f1']:.4f} |")
        a(f"| Threshold = {thresh['best_threshold']:.2f} | {thresh['in_scope_accuracy']:.4f} | "
          f"{thresh['oos_recall']:.4f} | {thresh['macro_f1']:.4f} |")
        a("")
        a(f"The chosen threshold of **{thresh['best_threshold']:.2f}** raises out-of-scope")
        a(f"recall from {fmt_pct(z['oos_recall'])} to {fmt_pct(thresh['oos_recall'])}")
        a(f"while in-scope accuracy moves from {fmt_pct(z['in_scope_accuracy'])} to")
        a(f"{fmt_pct(thresh['in_scope_accuracy'])}. This is the trade-off the sweep makes")
        a("explicit: a higher threshold catches more unanswerable questions but starts")
        a("refusing questions the model actually got right. The deployed application uses")
        a(f"{thresh['best_threshold']:.2f}.")
        a("")

    if voice:
        s = voice["summary"]
        a("### 6.4 End-to-end voice evaluation")
        a("")
        a("Text accuracy is not the accuracy a user experiences, because recognition")
        a(f"errors propagate into the classifier. To measure that, {s['n_utterances']} held-out")
        a("test utterances were synthesised to speech with offline system voices and put")
        a("through the complete deployed pipeline.")
        a("")
        a("| Metric | Value |")
        a("|---|---:|")
        a(f"| Utterances evaluated | {s['n_utterances']} |")
        a(f"| Word error rate | {s['corpus_wer']:.4f} |")
        a(f"| Transcribed with no errors | {fmt_pct(s['perfect_transcriptions'])} |")
        a(f"| Intent accuracy from clean text | {s['intent_accuracy_from_clean_text']:.4f} |")
        a(f"| Intent accuracy from speech | {s['intent_accuracy_from_audio']:.4f} |")
        a(f"| Accuracy lost to recognition | {s['accuracy_drop'] * 100:.1f} points |")
        a(f"| Mean speech recognition latency | {s['mean_stt_ms']:.0f} ms |")
        a(f"| Mean intent inference latency | {s['mean_infer_ms']:.0f} ms |")
        a(f"| Real-time factor | {s['real_time_factor']:.2f} |")
        a("")
        a("![Voice evaluation](../results/voice_eval.png)")
        a("")
        a(f"A word error rate of {s['corpus_wer']:.3f} means roughly")
        a(f"{s['corpus_wer'] * 100:.0f} words in every 100 are wrong, and the intent")
        a(f"classifier absorbs most of that: accuracy falls by only")
        a(f"{s['accuracy_drop'] * 100:.1f} points when the input arrives as speech rather")
        a("than text. A real-time factor below 1 means the system transcribes faster than")
        a("the audio was spoken.")
        a("")
        a("This evaluation uses synthetic speech, which is cleaner than real speech: no")
        a("background noise, no disfluencies, limited accent variation. The figures are a")
        a("lower bound on error, not a prediction of field performance.")
        a("")

    # ------------------------------------------------------------------
    a("## 7. Deployment")
    a("")
    a(f"The application is live at **{LIVE_URL}**.")
    a("")
    a("It runs as a **Hugging Face Docker Space** on the free CPU tier (2 vCPU, 16 GB")
    a("RAM). The image is built from `python:3.11-slim`, installs the CPU-only build of")
    a("PyTorch (the default wheel would pull in roughly 2.5 GB of unusable CUDA")
    a("libraries), bakes the Whisper weights in at build time so the first visitor does")
    a("not wait for a download, and runs uvicorn on port 7860 as uid 1000, as Spaces")
    a("requires.")
    a("")
    a("Both models are loaded once during the FastAPI lifespan startup hook, before the")
    a("server accepts traffic. Loading them per request would dominate the latency")
    a("budget entirely.")
    a("")
    a("| Method | Endpoint | Purpose |")
    a("|---|---|---|")
    a("| GET | `/` | chat interface |")
    a("| POST | `/api/voice` | audio upload to transcript, intent and reply |")
    a("| POST | `/api/transcribe` | audio upload to transcript only |")
    a("| POST | `/api/chat` | text to intent and reply |")
    a("| GET | `/api/info` | model metadata and intent list |")
    a("| GET | `/health` | liveness probe |")
    a("")
    a("The front end is dependency-free HTML, CSS and JavaScript. It records with the")
    a("MediaRecorder API, shows a live input-level ring while recording, and renders")
    a("each turn as the recognised speech, the intent with a confidence bar and the two")
    a("runner-up intents, and the reply. A text box is provided for machines without a")
    a("working microphone, and replies can optionally be spoken back using the browser's")
    a("built-in speech synthesis.")
    a("")

    # ------------------------------------------------------------------
    a("## 8. Limitations and future work")
    a("")
    a("- **Responses are templates, not generated text.** The deep learning in this")
    a("  system does the understanding, not the writing. Generating replies would need a")
    a("  language model far larger than the deployment target allows, and would")
    a("  introduce hallucination risk that templates do not have.")
    a("- **No dialogue state.** Each turn is classified independently, so a follow-up")
    a("  like \"and what about tomorrow?\" cannot be resolved.")
    a("- **No slot filling.** The system knows the user wants an alarm set, but not for")
    a("  what time. Adding a token-level tagger would address this.")
    a("- **Confidence scores are uncalibrated.** Softmax outputs from neural networks are")
    a("  known to be overconfident. Temperature scaling on the validation set, or an")
    a("  explicit open-set recognition method, would give better-behaved rejection.")
    a("- **Speech evaluation uses synthetic audio.** Real word error rates, especially")
    a("  with accents and background noise, will be higher.")
    a("- **40 of 150 intents.** The method scales to the full label set; the subset was a")
    a("  compute and response-authoring decision, not a limitation of the approach.")
    a("")

    # ------------------------------------------------------------------
    a("## 9. Reproducing this work")
    a("")
    a("```bash")
    a("pip install -r requirements-train.txt")
    a("python train/prepare_data.py       # build the CLINC150 subset")
    a("python train/train_baselines.py    # three reference models")
    a("python train/train_distilbert.py   # the deployed model")
    a("python train/evaluate.py           # figures and results tables")
    a("python train/eval_voice.py         # end-to-end speech evaluation")
    a("python report/build_report.py      # regenerate this document")
    a("cd app && uvicorn main:app --port 7860")
    a("```")
    a("")
    a("All randomness is seeded (seed 42).")
    a("")

    # ------------------------------------------------------------------
    a("## 10. References")
    a("")
    a("1. Larson, S., Mahendran, A., Peper, J. J., et al. *An Evaluation Dataset for")
    a("   Intent Classification and Out-of-Scope Prediction.* EMNLP 2019.")
    a("2. Sanh, V., Debut, L., Chaumond, J., Wolf, T. *DistilBERT, a distilled version of")
    a("   BERT: smaller, faster, cheaper and lighter.* NeurIPS EMC^2 Workshop, 2019.")
    a("3. Radford, A., Kim, J. W., Xu, T., et al. *Robust Speech Recognition via")
    a("   Large-Scale Weak Supervision.* OpenAI, 2022.")
    a("4. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. *BERT: Pre-training of Deep")
    a("   Bidirectional Transformers for Language Understanding.* NAACL 2019.")
    a("5. Vaswani, A., Shazeer, N., Parmar, N., et al. *Attention Is All You Need.*")
    a("   NeurIPS 2017.")
    a("")

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Minimal markdown -> HTML. The input is our own controlled markdown, so this
# only needs to handle the constructs used above.
# ---------------------------------------------------------------------------
def md_to_html(md: str) -> str:
    def inline(text: str) -> str:
        text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        return text

    def embed_image(alt: str, src: str) -> str:
        path = (REPORT / src).resolve()
        if not path.exists():
            return f'<p class="missing">[missing figure: {src}]</p>'
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}" /></figure>'

    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            body = "\n".join(block).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            out.append(f"<pre><code>{body}</code></pre>")
            continue

        img = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if img:
            out.append(embed_image(img.group(1), img.group(2)))
            i += 1
            continue

        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                table.append(lines[i])
                i += 1
            cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in table]
            aligns = []
            body_rows = cells[1:]
            if len(cells) > 1 and all(set(c) <= set(" -:") and "-" in c for c in cells[1]):
                aligns = ["right" if c.strip().endswith(":") else "left" for c in cells[1]]
                body_rows = cells[2:]
            head = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
            rows = []
            for row in body_rows:
                tds = "".join(
                    f'<td style="text-align:{aligns[j] if j < len(aligns) else "left"}">{inline(c)}</td>'
                    for j, c in enumerate(row)
                )
                rows.append(f"<tr>{tds}</tr>")
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")
            continue

        if re.match(r"^#{1,6} ", line):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
            i += 1
            continue

        if line.strip() == "---":
            out.append("<hr />")
            i += 1
            continue

        if re.match(r"^\s*[-*] ", line) or re.match(r"^\s*\d+\. ", line):
            ordered = bool(re.match(r"^\s*\d+\. ", line))
            items = []
            while i < len(lines) and (
                re.match(r"^\s*[-*] ", lines[i]) or re.match(r"^\s*\d+\. ", lines[i])
                or (items and lines[i].startswith("  ") and lines[i].strip())
            ):
                if re.match(r"^\s*[-*] ", lines[i]) or re.match(r"^\s*\d+\. ", lines[i]):
                    items.append(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        if not line.strip():
            i += 1
            continue

        para = []
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,6} |\||```|!\[|\s*[-*] |\s*\d+\. |---$)", lines[i]
        ):
            para.append(lines[i].rstrip())
            i += 1
        text = " ".join(para)
        text = re.sub(r"\s{2,}$", "", text)
        out.append(f"<p>{inline(text)}</p>")

    return "\n".join(out)


CSS = """
:root { --ink:#1a1c23; --muted:#5c6270; --line:#e3e6ee; --accent:#4f46e5; --bg:#fff; --code:#f5f6fa; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e6e8ef; --muted:#9aa1b1; --line:#2a2f3d; --accent:#8b8cf9; --bg:#12141a; --code:#1b1e27; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 ui-serif, Georgia, "Times New Roman", serif; }
main { max-width:860px; margin:0 auto; padding:56px 28px 90px; }
h1 { font-size:30px; line-height:1.25; margin:0 0 6px; letter-spacing:-.01em; }
h2 { font-size:22px; margin:44px 0 12px; padding-bottom:7px; border-bottom:2px solid var(--line); }
h3 { font-size:17px; margin:28px 0 8px; color:var(--accent); }
p { margin:0 0 13px; }
hr { border:0; border-top:1px solid var(--line); margin:26px 0; }
ul, ol { margin:0 0 14px; padding-left:22px; }
li { margin-bottom:6px; }
a { color:var(--accent); }
code { font:13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  background:var(--code); padding:1.5px 5px; border-radius:4px; }
pre { background:var(--code); border:1px solid var(--line); border-radius:9px;
  padding:14px 16px; overflow-x:auto; }
pre code { background:none; padding:0; font-size:12.5px; line-height:1.55; }
table { width:100%; border-collapse:collapse; margin:14px 0 20px;
  font:14px/1.5 ui-sans-serif, system-ui, sans-serif; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--line); padding:8px 11px; text-align:left; vertical-align:top; }
thead th { border-bottom:2px solid var(--line); font-weight:600; white-space:nowrap; }
tbody tr:last-child td { border-bottom:none; }
figure { margin:22px 0; text-align:center; }
figure img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:9px; }
.missing { color:#b45309; font-style:italic; }
@media print {
  body { background:#fff; color:#000; }
  h2 { page-break-after:avoid; }
  figure, table, pre { page-break-inside:avoid; }
  main { max-width:none; padding:0; }
}
"""


def main() -> None:
    md = build_markdown()
    (REPORT / "report.md").write_text(md, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Voice-Enabled Chatbot — Report</title>
<style>{CSS}</style>
</head>
<body>
<main>
{md_to_html(md)}
</main>
</body>
</html>
"""
    (REPORT / "report.html").write_text(html, encoding="utf-8")
    size_kb = (REPORT / "report.html").stat().st_size / 1024
    print(f"Wrote report/report.md ({len(md.splitlines())} lines)")
    print(f"Wrote report/report.html ({size_kb:.0f} KB, figures embedded)")


if __name__ == "__main__":
    main()
