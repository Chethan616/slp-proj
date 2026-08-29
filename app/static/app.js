/* Voice-enabled chatbot front end.
 *
 * Records microphone audio with MediaRecorder, posts it to /api/voice, and
 * renders the recognised speech, the predicted intent with its confidence, and
 * the generated reply. Falls back to a text box when there is no microphone.
 */

const transcriptEl = document.getElementById("transcript");
const emptyState = document.getElementById("emptyState");
const micBtn = document.getElementById("micBtn");
const micLabel = document.getElementById("micLabel");
const timerEl = document.getElementById("timer");
const levelEl = document.getElementById("level");
const textForm = document.getElementById("textForm");
const textInput = document.getElementById("textInput");
const speakToggle = document.getElementById("speakToggle");
const modelInfoEl = document.getElementById("modelInfo");
const suggestionsEl = document.getElementById("suggestions");

const SUGGESTIONS = [
  "What's the weather like today?",
  "Set an alarm for 7 am",
  "What is my account balance?",
  "Tell me a joke",
  "How do I get to the airport?",
  "What can I ask you?",
];

let mediaRecorder = null;
let chunks = [];
let stream = null;
let audioCtx = null;
let analyser = null;
let rafId = null;
let timerId = null;
let startedAt = 0;
let busy = false;

/* ---------------- pipeline indicator ---------------- */

function setStage(name) {
  const stages = ["record", "stt", "nlu", "reply"];
  const idx = stages.indexOf(name);
  document.querySelectorAll(".stage").forEach((el) => {
    const i = stages.indexOf(el.dataset.stage);
    el.classList.toggle("active", i === idx);
    el.classList.toggle("done", idx >= 0 && i < idx);
  });
}

function clearStages() {
  document.querySelectorAll(".stage").forEach((el) => el.classList.remove("active", "done"));
}

/* ---------------- rendering ---------------- */

function hideEmptyState() {
  if (emptyState && emptyState.parentNode) emptyState.remove();
}

function scrollDown() {
  transcriptEl.parentElement.scrollTop = transcriptEl.parentElement.scrollHeight;
}

function addThinking(message) {
  hideEmptyState();
  const el = document.createElement("div");
  el.className = "thinking";
  el.textContent = message;
  transcriptEl.appendChild(el);
  scrollDown();
  return el;
}

function addError(message) {
  hideEmptyState();
  const el = document.createElement("div");
  el.className = "error";
  el.textContent = message;
  transcriptEl.appendChild(el);
  scrollDown();
}

function renderTurn(data) {
  hideEmptyState();

  const turn = document.createElement("div");
  turn.className = "turn";

  // 1. what the speech recogniser heard
  const user = document.createElement("div");
  user.className = "bubble user";
  const sttMeta =
    data.source === "voice"
      ? `Whisper · ${data.stt_ms} ms · ${data.audio_seconds}s of audio`
      : "typed input";
  user.innerHTML = `
    <span class="label">Recognised speech</span>
    <div class="text"></div>
    <div class="meta"><span>${sttMeta}</span></div>`;
  user.querySelector(".text").textContent = data.transcript || "(nothing recognised)";
  turn.appendChild(user);

  // 2. what the intent model decided
  if (data.top && data.top.length) {
    const pct = Math.round(data.confidence * 100);
    const low = data.below_threshold || data.intent === "oos";
    const alts = data.top
      .slice(1)
      .map((t) => `${t.intent} ${Math.round(t.confidence * 100)}%`)
      .join(" · ");

    const row = document.createElement("div");
    row.className = "intent-row";
    row.innerHTML = `
      <span>Intent</span>
      <span class="chip ${low ? "oos" : ""}"></span>
      <span class="bar ${low ? "low" : ""}"><i style="width:${pct}%"></i></span>
      <span>${pct}%</span>
      <span class="alt">${alts ? "next: " + alts : ""}</span>
      <span>· ${data.infer_ms} ms</span>`;
    row.querySelector(".chip").textContent = data.intent;
    turn.appendChild(row);
  }

  // 3. the reply
  const bot = document.createElement("div");
  bot.className = "bubble bot";
  bot.innerHTML = `<span class="label">Vox</span><div class="text"></div>`;
  bot.querySelector(".text").textContent = data.reply;
  turn.appendChild(bot);

  transcriptEl.appendChild(turn);
  scrollDown();

  if (speakToggle.checked && data.reply && "speechSynthesis" in window) {
    const utterance = new SpeechSynthesisUtterance(data.reply);
    utterance.rate = 1.02;
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  }
}

/* ---------------- networking ---------------- */

async function postVoice(blob) {
  const form = new FormData();
  const ext = (blob.type.split("/")[1] || "webm").split(";")[0];
  form.append("audio", blob, `speech.${ext}`);
  const res = await fetch("/api/voice", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `server returned ${res.status}`);
  }
  return res.json();
}

async function postText(text) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `server returned ${res.status}`);
  }
  return res.json();
}

/* ---------------- recording ---------------- */

function drawLevel() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  let peak = 0;
  for (const v of data) peak = Math.max(peak, Math.abs(v - 128) / 128);
  levelEl.style.transform = `scale(${1 + Math.min(peak, 1) * 0.45})`;
  levelEl.style.opacity = String(0.35 + Math.min(peak, 1) * 0.65);
  rafId = requestAnimationFrame(drawLevel);
}

function tickTimer() {
  const secs = (Date.now() - startedAt) / 1000;
  timerEl.textContent = secs.toFixed(1) + "s";
  // Hard stop so a forgotten open mic cannot upload a huge file.
  if (secs > 30) stopRecording();
}

async function startRecording() {
  if (busy) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
  } catch (err) {
    addError(
      "Microphone access was blocked. Allow the mic in your browser, or type your message below."
    );
    return;
  }

  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRecorder.onstop = handleRecordingStopped;
  mediaRecorder.start();

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  drawLevel();

  startedAt = Date.now();
  timerId = setInterval(tickTimer, 100);

  micBtn.classList.add("recording");
  micBtn.setAttribute("aria-label", "Stop recording");
  micLabel.textContent = "Listening — click to stop";
  setStage("record");
}

function teardownAudio() {
  if (rafId) cancelAnimationFrame(rafId);
  if (timerId) clearInterval(timerId);
  rafId = timerId = null;
  if (audioCtx) audioCtx.close().catch(() => {});
  audioCtx = analyser = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  levelEl.style.transform = "scale(1)";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
}

async function handleRecordingStopped() {
  const elapsed = (Date.now() - startedAt) / 1000;
  teardownAudio();
  micBtn.classList.remove("recording");
  micBtn.setAttribute("aria-label", "Start recording");
  timerEl.textContent = "";

  const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });

  if (elapsed < 0.4 || blob.size < 1200) {
    micLabel.textContent = "Click to speak";
    addError("That recording was too short — hold on a moment longer and speak clearly.");
    clearStages();
    return;
  }

  busy = true;
  micBtn.disabled = true;
  micLabel.textContent = "Processing…";
  setStage("stt");
  const thinking = addThinking("Transcribing and classifying");

  try {
    const data = await postVoice(blob);
    setStage("reply");
    thinking.remove();
    renderTurn(data);
  } catch (err) {
    thinking.remove();
    addError("Could not process that audio: " + err.message);
  } finally {
    busy = false;
    micBtn.disabled = false;
    micLabel.textContent = "Click to speak";
    setTimeout(clearStages, 900);
  }
}

/* ---------------- events ---------------- */

micBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") stopRecording();
  else startRecording();
});

// Space bar as push-to-talk, as long as focus is not in the text box.
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  if (document.activeElement === textInput) return;
  e.preventDefault();
  if (!(mediaRecorder && mediaRecorder.state === "recording")) startRecording();
});

document.addEventListener("keyup", (e) => {
  if (e.code !== "Space") return;
  if (document.activeElement === textInput) return;
  stopRecording();
});

textForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text || busy) return;
  textInput.value = "";
  busy = true;
  setStage("nlu");
  const thinking = addThinking("Classifying");
  try {
    const data = await postText(text);
    setStage("reply");
    thinking.remove();
    renderTurn(data);
  } catch (err) {
    thinking.remove();
    addError("Request failed: " + err.message);
  } finally {
    busy = false;
    setTimeout(clearStages, 900);
  }
});

/* ---------------- init ---------------- */

SUGGESTIONS.forEach((text) => {
  const li = document.createElement("li");
  li.textContent = text;
  li.addEventListener("click", () => {
    textInput.value = text;
    textForm.requestSubmit();
  });
  suggestionsEl.appendChild(li);
});

if (!navigator.mediaDevices || !window.MediaRecorder) {
  micBtn.disabled = true;
  micLabel.textContent = "Microphone not supported — use the text box";
}

fetch("/api/info")
  .then((r) => r.json())
  .then((info) => {
    modelInfoEl.textContent = `${info.stt_model} → ${info.intent_model} · ${info.num_intents} intents`;
  })
  .catch(() => {});
