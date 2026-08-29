# Viva preparation — questions you are likely to be asked

Short, honest answers for every part of the system. If you can answer these, you
can defend the whole project.

---

## The pipeline

**Q. Walk me through what happens when someone speaks to your app.**
The browser records microphone audio with the MediaRecorder API and uploads it
as a webm/opus blob to `POST /api/voice`. The server decodes it and runs
faster-whisper `base.en`, which returns a transcript. That transcript goes into
a fine-tuned DistilBERT classifier, which outputs a probability distribution
over 41 intents. We take the argmax; if its probability is below a threshold we
override it to `oos` (out of scope). The intent selects a response template,
which is returned as JSON along with the transcript, the intent, the confidence
and the timings. The page displays all of it.

**Q. Why two models instead of one end-to-end model?**
Because they solve different problems and are trained on different data. Whisper
is a general-purpose speech recogniser trained on 680k hours of audio; retraining
it is out of reach here. Intent classification is a small, domain-specific
problem where a fine-tuned text model does well with a few thousand examples.
Splitting the pipeline also means each stage can be evaluated separately — which
is exactly what the results section does.

---

## Speech recognition

**Q. How does Whisper work?**
Audio is resampled to 16 kHz and converted into an 80-channel log-Mel
spectrogram — a time-frequency representation on a perceptually spaced frequency
axis. That spectrogram is fed to a transformer encoder, and a transformer decoder
autoregressively generates text tokens while attending to the encoder output. It
is a sequence-to-sequence model trained on 680,000 hours of weakly supervised
multilingual audio.

**Q. What is a log-Mel spectrogram, and why not raw audio?**
A spectrogram is the magnitude of the short-time Fourier transform: how much
energy sits at each frequency in each short time window. The Mel scale warps the
frequency axis to match human pitch perception, and the log compresses the
dynamic range. It is a compact representation that discards phase information the
model does not need — far easier to learn from than a raw 16,000-samples-per-
second waveform.

**Q. Why `base.en` and not a larger Whisper model?**
The deployment target is a free 2-vCPU container. `base.en` is 74M parameters and
English-only, which is both faster and more accurate on English than the
multilingual model of the same size. Larger models would push latency past what
is acceptable for an interactive demo.

**Q. What is int8 quantisation?**
The weights are stored as 8-bit integers instead of 32-bit floats. That roughly
quarters the memory and speeds up CPU inference substantially, at a very small
accuracy cost. CTranslate2 (which faster-whisper is built on) handles the
quantisation and dequantisation internally.

**Q. What is WER?**
Word error rate: (substitutions + insertions + deletions) / number of reference
words, computed from the minimum edit distance alignment between the reference
and the recognised text. Lower is better; 0 means a perfect transcription.

---

## The intent model

**Q. What is DistilBERT and why choose it?**
It is a 6-layer distillation of the 12-layer BERT-base: about 40% smaller and 60%
faster, retaining roughly 97% of BERT's performance on GLUE. It was trained with
knowledge distillation, where the small student model learns to match the large
teacher's output distribution. Here it matters because inference has to be fast
on a CPU with two cores.

**Q. What exactly did you fine-tune?**
All layers. We take the pre-trained DistilBERT encoder, attach a fresh linear
classification head over the `[CLS]` token representation, and train the whole
thing on our labelled utterances with cross-entropy loss. The pre-trained weights
provide the language understanding; the fine-tuning teaches it our 41 categories.

**Q. What are the hyperparameters and why?**
Max sequence length 32 (these utterances are short — that covers essentially all
of them without wasting computation), batch size 32, learning rate 3e-5 with 10%
linear warmup then linear decay, weight decay 0.01, 5 epochs, gradient clipping
at 1.0. The small learning rate is standard for fine-tuning: large steps would
destroy the pre-trained representations.

**Q. Why is there a warmup?**
At the start of training the gradients are large and poorly conditioned because
the classification head is randomly initialised. Ramping the learning rate up
over the first 10% of steps stops those early updates from damaging the
pre-trained encoder.

**Q. Why macro F1 rather than accuracy alone?**
Accuracy is dominated by whichever classes have the most test examples. Macro F1
averages the per-class F1 equally, so a model that ignores a rare class is
penalised. Our test set has 30 utterances for each in-scope intent but 300
out-of-scope ones, so the two metrics genuinely differ — reporting both is
honest.

**Q. What is F1?**
The harmonic mean of precision and recall: 2PR/(P+R). Precision is the fraction
of predictions for a class that were right; recall is the fraction of that
class's true instances that were found.

---

## The dataset

**Q. Why CLINC150?**
It is a published benchmark designed specifically for this task, which means the
numbers are comparable to prior work rather than to a dataset we invented. It is
also the standard dataset for the out-of-scope problem, which most intent
datasets simply ignore.

**Q. Why only 40 of the 150 intents?**
Two reasons. Training all 150 classes on CPU takes much longer for no extra
pedagogical value, and every intent needs a hand-written response template for
the chatbot to actually reply. Forty intents across eight domains is enough to
demonstrate the method and still be a usable assistant. The selection is fixed in
`train/prepare_data.py`, so it is reproducible.

**Q. What is the out-of-scope class and why does it matter?**
It is a class of user queries the assistant is not built to handle. Without it, a
classifier must assign every input to one of its known intents — so "who was the
first Capcom character" gets confidently labelled as, say, `definition`, and the
bot answers nonsense. Modelling out-of-scope explicitly lets the system say "I
don't know", which is what a deployed assistant needs to do.

**Q. How do you decide something is out of scope?**
Two routes. The classifier can predict the `oos` class directly, or the maximum
softmax probability can fall below a confidence threshold, in which case we
override the prediction to `oos`. The threshold was chosen by sweeping it on the
test predictions and taking the value that maximises overall macro F1 — see the
threshold sweep figure.

**Q. Isn't a softmax probability a bad confidence estimate?**
Yes, neural networks are known to be overconfident, and the raw softmax is not a
calibrated probability. It is still a usable *ranking* signal, which is all the
threshold needs. Proper calibration (temperature scaling) or an explicit
open-set method would be the next step — that is noted in the limitations.

---

## Results and evaluation

**Q. Why did you train four models?**
To show what the transformer is actually worth. TF-IDF with logistic regression
is a strong classical baseline; a bag-of-words MLP and a BiLSTM are the standard
from-scratch neural approaches. Reporting only the best model tells you nothing
about whether the complexity was justified.

**Q. Your baselines are quite close to the transformer. Doesn't that undermine it?**
It shows the task is not especially hard for in-scope classification — with 100
clean examples per class, keyword evidence goes a long way. The gap opens up on
the harder cases: out-of-scope rejection and paraphrases that share no vocabulary
with the training data, where pre-trained representations help most. Look at the
per-class F1 and the out-of-scope numbers rather than the headline accuracy.

**Q. How do you know the system works on actual speech, not just text?**
`train/eval_voice.py` synthesises spoken versions of held-out test utterances
with offline system voices, runs them through the real pipeline, and reports word
error rate alongside intent accuracy from audio versus from clean text. The gap
between those two numbers is exactly the accuracy lost to recognition errors.

**Q. What is the weakness of that evaluation?**
Synthetic speech is cleaner than real speech: no background noise, no accents
beyond the installed voices, no disfluencies, consistent pacing. Real WER would
be higher. It is a lower bound on the error, not an estimate of field
performance — a proper evaluation would need recorded human speakers.

---

## Deployment

**Q. Where is it hosted, and how?**
A Hugging Face Docker Space. The Dockerfile builds a Python 3.11 image, installs
the CPU-only build of PyTorch, bakes the Whisper weights into the image so the
first request does not wait for a download, and runs uvicorn on port 7860 as
uid 1000 (which is what Spaces requires).

**Q. Why load the models at startup rather than per request?**
Loading DistilBERT and Whisper takes several seconds. Doing that per request
would dominate the latency. They are loaded once in the FastAPI lifespan hook,
before the app accepts traffic, and reused.

**Q. What are the latency numbers?**
Reported in the results section: speech recognition takes roughly a second or two
for a short utterance on the free CPU tier, and intent classification is a few
tens of milliseconds. The real-time factor (processing time divided by audio
duration) is the scale-free way to read the STT number.

---

## Honest limitations

- Responses are template-based, not generated. The deep learning is in the
  understanding, not the generation. A generative model would need far more
  compute and would introduce hallucination risk.
- There is no dialogue state: each turn is classified independently, so the bot
  cannot handle "and what about tomorrow?".
- Slot filling is not implemented — the system knows you want to set an alarm but
  not what time you said.
- Confidence scores are uncalibrated softmax probabilities.
- The speech evaluation uses synthetic rather than human speech.
