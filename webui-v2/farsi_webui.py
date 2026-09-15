import gradio as gr
from pocket_tts import TTSModel
import numpy as np
import re
import torch
import threading
from queue import Queue
from transformers import AutoTokenizer, T5ForConditionalGeneration
from normalize_fa import normalize_for_model


# ============================================================
#  Load TTS model — v2
# ============================================================

model = TTSModel.load_model(
    config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml",
    temp=0.3
)

voice_path = "example_voice.wav"
voice_state = model.get_state_for_audio_prompt(voice_path)


# ============================================================
#  Load G2P model
# ============================================================

G2P_REPO = "mehdi-hf/Homo-GE2PE-Persian-HF"
g2p_tok = AutoTokenizer.from_pretrained(G2P_REPO)
g2p_model = T5ForConditionalGeneration.from_pretrained(G2P_REPO).eval()

TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

YIELD_INTERVAL_SEC = 0.5


# ============================================================
#  Chunking thresholds
# ============================================================

# Tier 1: sentence boundaries (always applied)
# Tier 2: commas / semicolons (applied when sentence > MAX_CHARS_PER_CHUNK)
# Tier 3: conjunctions (applied when tiers 1+2 fail)
# Tier 4: verbs (last resort)

MAX_CHARS_PER_CHUNK = 150
MIN_CONJ_SPLITS = 1
MIN_VERB_SPLITS = 1


# ============================================================
#  Conjunctions (fa.wikipedia.org/wiki/حرف_ربط)
# ============================================================

_CONJUNCTIONS = [
    "به شرط آنکه", "به‌شرط آنکه",
    "از آنجا که", "ازآنجا که",
    "از این رو", "ازاین‌رو",
    "با این حال", "بااین‌حال",
    "با اینکه", "بااینکه",
    "همین که", "همینکه",
    "زیرا که", "زیراکه",
    "چون که", "چونکه",
    "اگر چه", "اگرچه",
    "چنان که", "چنانکه",
    "چنان چه", "چنانچه",
    "بدان که", "بدانکه",
    "و", "یا", "پس", "اگر", "نه", "چون", "اما",
    "خواه", "زیرا", "لیکن", "ولی", "بلکه",
]

_CONJ_SORTED = sorted(_CONJUNCTIONS, key=lambda s: len(s.split()), reverse=True)


# ============================================================
#  Persian verbs (SOV clause-end markers)
# ============================================================

_VERB_PHRASES = [
    "شده است", "شده بود", "شده‌اند", "شده بودند",
    "نشده است", "نشده بود",
    "کرده است", "کرده بود", "کرده‌اند", "کرده بودند",
    "نکرده است", "نکرده بود",
    "رفته است", "رفته بود", "رفته‌اند", "رفته بودند",
    "آمده است", "آمده بود", "آمده‌اند",
    "داده است", "داده بود", "داده‌اند",
    "گرفته است", "گرفته بود",
    "گفته است", "گفته بود",
    "دیده است", "دیده بود",
    "خورده است", "خورده بود",
    "مانده است", "مانده بود",
    "خواسته است", "خواسته بود",
    "توانسته است", "توانسته بود",
]

_VERB_WORDS = {
    "است", "هست", "نیست", "بود", "نبود", "باشد", "نباشد",
    "هستند", "نیستند", "بودند", "نبودند", "باشند", "نباشند",
    "هستم", "نیستم", "بودم", "نبودم", "باشم", "نباشم",
    "هستی", "نیستی", "بودی", "نبودی", "باشی", "نباشی",
    "هستیم", "نیستیم", "بودیم", "نبودیم", "باشیم", "نباشیم",
    "هستید", "نیستید", "بودید", "نبودید", "باشید", "نباشید",
    "شد", "نشد", "شده", "نشده",
    "شدم", "شدی", "شدیم", "شدید", "شدند",
    "می‌شود", "نمی‌شود", "می‌شوند", "نمی‌شوند",
    "می‌شد", "نمی‌شد", "می‌شدند", "نمی‌شدند",
    "بشود", "بشوند",
    "کرد", "نکرد", "کرده", "نکرده",
    "کردم", "کردی", "کردیم", "کردید", "کردند",
    "می‌کند", "نمی‌کند", "می‌کنند", "نمی‌کنند",
    "می‌کرد", "نمی‌کرد", "می‌کردند", "نمی‌کردند",
    "بکند", "بکنند",
    "داد", "نداد", "داده", "نداده",
    "دادم", "دادی", "دادیم", "دادید", "دادند",
    "می‌دهد", "نمی‌دهد", "می‌دهند", "نمی‌دهند",
    "می‌داد", "نمی‌داد", "بدهد", "بدهند",
    "گرفت", "نگرفت", "گرفته", "نگرفته",
    "گرفتم", "گرفتی", "گرفتیم", "گرفتید", "گرفتند",
    "می‌گیرد", "نمی‌گیرد", "می‌گیرند", "نمی‌گیرند",
    "می‌گرفت", "بگیرد", "بگیرند",
    "رفت", "نرفت", "رفته", "نرفته",
    "رفتم", "رفتی", "رفتیم", "رفتید", "رفتند",
    "می‌رود", "نمی‌رود", "می‌روند", "نمی‌روند",
    "می‌رفت", "برود", "بروند",
    "آمد", "نیامد", "آمده", "نیامده",
    "آمدم", "آمدی", "آمدیم", "آمدید", "آمدند",
    "می‌آید", "نمی‌آید", "می‌آیند", "نمی‌آیند",
    "می‌آمد", "بیاید", "بیایند",
    "گفت", "نگفت", "گفته", "نگفته",
    "گفتم", "گفتی", "گفتیم", "گفتید", "گفتند",
    "می‌گوید", "نمی‌گوید", "می‌گویند",
    "می‌گفت", "بگوید", "بگویند",
    "دید", "ندید", "دیده", "ندیده",
    "دیدم", "دیدی", "دیدیم", "دیدند",
    "می‌بیند", "نمی‌بیند", "می‌بینند",
    "می‌دید", "ببیند", "ببینند",
    "خورد", "نخورد", "خورده", "نخورده",
    "خوردم", "خوردی", "خوردند",
    "می‌خورد", "نمی‌خورد", "می‌خورند",
    "بخورد", "بخورند",
    "ماند", "نماند", "مانده", "نمانده",
    "ماندم", "ماندند",
    "می‌ماند", "نمی‌ماند", "بماند", "بمانند",
    "خواست", "نخواست", "خواسته", "نخواسته",
    "خواستم", "خواستند",
    "می‌خواهد", "نمی‌خواهد", "می‌خواهند",
    "بخواهد", "بخواهند",
    "توانست", "نتوانست", "توانسته", "نتوانسته",
    "می‌تواند", "نمی‌تواند", "می‌توانند", "نمی‌توانند",
    "بتواند", "بتوانند",
    "رسید", "نرسید", "رسیده", "می‌رسد", "برسد",
    "افتاد", "افتاده", "می‌افتد", "بیفتد",
    "نشست", "نشسته", "می‌نشیند", "بنشیند",
    "ایستاد", "ایستاده", "می‌ایستد", "بایستد",
    "برگشت", "برگشته", "برمی‌گردد", "برگردد",
    "مرد", "مرده", "می‌میرد", "بمیرد",
    "خرید", "خریده", "می‌خرد", "بخرد",
    "فروخت", "فروخته", "می‌فروشد", "بفروشد",
    "نوشت", "نوشته", "می‌نویسد", "بنویسد",
    "خواند", "خوانده", "می‌خواند", "بخواند",
    "شنید", "شنیده", "می‌شنود", "بشنود",
    "دانست", "دانسته", "می‌داند", "بداند",
    "فهمید", "فهمیده", "می‌فهمد", "بفهمد",
}

_VERB_PHRASES_SORTED = sorted(_VERB_PHRASES, key=lambda s: len(s.split()), reverse=True)

_NO_SPLIT_BEFORE = {
    "را", "به", "از", "با", "در", "بر", "برای", "بدون",
    "توسط", "نزد", "پیش", "روی", "زیر", "بالای", "کنار", "بین", "میان",
}


# ============================================================
#  G2P
# ============================================================

def phonemise(text: str) -> str:
    text = normalize_for_model(text)
    text = text.replace("؟", "").replace("?", "")
    enc = g2p_tok([text], add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = g2p_model.generate(
            **enc, num_beams=5, max_length=512, early_stopping=True
        )
    raw = g2p_tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    return raw.replace("1", "").translate(TO_PHONEMES)


# ============================================================
#  Audio helper
# ============================================================

def to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


# ============================================================
#  Chunking — three tiers on Persian text
# ============================================================

def split_persian_sentences(text: str):
    text = normalize_for_model(text)
    text = re.sub(r'([.?!؛،])(\S)', r'\1 \2', text)
    sentences = re.split(r'(?<=[.?!؛؟])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _find_conjunction_indices(words):
    indices = []
    i = 0
    while i < len(words):
        for conj in _CONJ_SORTED:
            cw = conj.split()
            n = len(cw)
            if i + n <= len(words) and all(words[i + j] == cw[j] for j in range(n)):
                indices.append(i)
                break
        i += 1
    return indices


def split_sentence_at_conjunctions(sentence: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    if len(sentence) <= max_chars:
        return [sentence]
    words = sentence.split()
    conj_indices = _find_conjunction_indices(words)
    if len(conj_indices) < MIN_CONJ_SPLITS:
        return [sentence]

    segments = []
    prev = 0
    for idx in conj_indices:
        if idx > prev:
            seg = " ".join(words[prev:idx]).strip()
            if seg:
                segments.append(seg)
        prev = idx
    tail = " ".join(words[prev:]).strip()
    if tail:
        segments.append(tail)
    if len(segments) <= 1:
        return [sentence]

    chunks, current = [], segments[0]
    for seg in segments[1:]:
        candidate = current + " " + seg
        if len(candidate) > max_chars and current.strip():
            chunks.append(current.strip())
            current = seg
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _find_verb_end_indices(words):
    end_indices = []
    i = 0
    while i < len(words):
        matched_len = 0
        for phrase in _VERB_PHRASES_SORTED:
            pw = phrase.split()
            n = len(pw)
            if i + n <= len(words) and all(words[i + j] == pw[j] for j in range(n)):
                matched_len = n
                break
        if matched_len == 0 and words[i] in _VERB_WORDS:
            matched_len = 1
        if matched_len > 0:
            end_indices.append(i + matched_len - 1)
            i += matched_len
        else:
            i += 1
    return end_indices


def split_sentence_at_verbs(sentence: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    if len(sentence) <= max_chars:
        return [sentence]
    words = sentence.split()
    verb_end_indices = _find_verb_end_indices(words)
    if len(verb_end_indices) < MIN_VERB_SPLITS:
        return [sentence]

    split_points = []
    for idx in verb_end_indices:
        if idx + 1 < len(words) and words[idx + 1] in _NO_SPLIT_BEFORE:
            continue
        split_points.append(idx)
    if not split_points:
        return [sentence]

    segments, prev = [], 0
    for idx in split_points:
        seg = " ".join(words[prev:idx + 1]).strip()
        if seg:
            segments.append(seg)
        prev = idx + 1
    tail = " ".join(words[prev:]).strip()
    if tail:
        segments.append(tail)
    if len(segments) <= 1:
        return [sentence]

    chunks, current = [], segments[0]
    for seg in segments[1:]:
        candidate = current + " " + seg
        if len(candidate) > max_chars and current.strip():
            chunks.append(current.strip())
            current = seg
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunks(sentences):
    chunks = []
    for sentence in sentences:
        if len(sentence) <= MAX_CHARS_PER_CHUNK:
            chunks.append(sentence)
            continue
        conj_chunks = split_sentence_at_conjunctions(sentence)
        if len(conj_chunks) > 1:
            chunks.extend(conj_chunks)
            continue
        chunks.extend(split_sentence_at_verbs(sentence))
    return chunks


# ============================================================
#  Streaming synthesis (threaded G2P, v1-style chunking on top)
# ============================================================

def synthesize_streaming(text):
    if not text or not text.strip():
        yield None
        return

    sample_rate = model.sample_rate
    yield_every = int(sample_rate * YIELD_INTERVAL_SEC)

    sentences = split_persian_sentences(text)
    chunks = build_chunks(sentences)
    if not chunks:
        yield None
        return

    phoneme_queue = Queue()

    def g2p_worker():
        for idx, c in enumerate(chunks):
            try:
                phoneme_queue.put((idx, phonemise(c)))
            except Exception as e:
                print(f"G2P failed on chunk {idx}: {e}")
                phoneme_queue.put((idx, ""))

    worker = threading.Thread(target=g2p_worker, daemon=True)
    worker.start()

    pending_audio = np.zeros(0, dtype=np.float32)
    samples_since_yield = 0

    for _ in range(len(chunks)):
        idx, phonemes = phoneme_queue.get()
        if not phonemes.strip():
            continue
        print(f"Chunk {idx+1}/{len(chunks)}: {phonemes[:60]}...")

        try:
            stream = model.generate_audio_stream(
                voice_state, phonemes, frames_after_eos=0
            )
        except TypeError:
            stream = model.generate_audio_stream(voice_state, phonemes)

        for frame in stream:
            frame_np = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
            frame_np = frame_np.astype(np.float32).reshape(-1)
            pending_audio = np.concatenate([pending_audio, frame_np])
            samples_since_yield += len(frame_np)
            if samples_since_yield >= yield_every:
                yield (sample_rate, to_int16(pending_audio))
                pending_audio = np.zeros(0, dtype=np.float32)
                samples_since_yield = 0

        silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
        pending_audio = np.concatenate([pending_audio, silence])
        if len(pending_audio) > 0:
            yield (sample_rate, to_int16(pending_audio))
            pending_audio = np.zeros(0, dtype=np.float32)
            samples_since_yield = 0

    if len(pending_audio) > 0:
        yield (sample_rate, to_int16(pending_audio))


# ============================================================
#  Gradio UI
# ============================================================

with gr.Blocks(title="Pocket TTS - Farsi v2 (Chunked)") as iface:
    gr.Markdown(
        "## Pocket TTS - Farsi v2 (Persian) — Chunked Streaming\n"
        "Paste Persian text and press **Generate**. Long sentences are "
        "split at conjunctions and verbs before phonemisation. "
        "Press **Stop** to cancel mid-generation."
    )
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(
                label="Persian Text",
                lines=10,
                placeholder="متن طولانی خود را اینجا وارد کنید..."
            )
            with gr.Row():
                btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            clear = gr.ClearButton([txt], value="Clear Text")
        with gr.Column():
            out_audio = gr.Audio(
                label="Generated Speech",
                type="numpy",
                autoplay=True,
                streaming=True,
            )

    gen_event = btn.click(fn=synthesize_streaming, inputs=txt, outputs=out_audio)
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

iface.queue().launch(server_name="127.0.0.1", server_port=7862)
