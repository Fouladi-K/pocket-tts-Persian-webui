import gradio as gr
from pocket_tts import TTSModel
import numpy as np
import re
import pocket_tts.default_parameters as dp

# Override the global token limit before loading the model
dp.MAX_TOKEN_PER_CHUNK = 200

# Load the model (CPU by default)
model = TTSModel.load_model(
    config="hf://mehdi-hf/pocket-tts-farsi/farsi.yaml",
    temp=0.3
)

# Load the default voice prompt
voice_path = "example_voice.wav"
voice_state = model.get_state_for_audio_prompt(voice_path)

# How often to push new audio to the UI (in seconds of audio)
YIELD_INTERVAL_SEC = 0.5

# ---- Chunking thresholds ----
# Tier 1: sentence splitter (punctuation) — always on
# Tier 2: conjunction splitter — only when sentence > MAX_CHARS_PER_CHUNK
# Tier 3: verb splitter — only when tiers 1+2 fail

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
#  Persian verbs (used as clause-end markers — SOV language)
# ============================================================

# Multi-word / perfect forms — matched as a unit, checked first
_VERB_PHRASES = [
    # شده است family
    "شده است", "شده بود", "شده‌اند", "شده بودند",
    "نشده است", "نشده بود",
    # کرده است family
    "کرده است", "کرده بود", "کرده‌اند", "کرده بودند",
    "نکرده است", "نکرده بود",
    # رفته است family
    "رفته است", "رفته بود", "رفته‌اند", "رفته بودند",
    # آمده است family
    "آمده است", "آمده بود", "آمده‌اند",
    # داده است family
    "داده است", "داده بود", "داده‌اند",
    # گرفته است family
    "گرفته است", "گرفته بود",
    # گفته است family
    "گفته است", "گفته بود",
    # دیده است family
    "دیده است", "دیده بود",
    # خورده است family
    "خورده است", "خورده بود",
    # مانده است family
    "مانده است", "مانده بود",
    # خواسته است family
    "خواسته است", "خواسته بود",
    # توانسته است family
    "توانسته است", "توانسته بود",
]

# Single-word verb forms
_VERB_WORDS = {
    # Copulas & auxiliaries (بودن)
    "است", "هست", "نیست", "بود", "نبود", "باشد", "نباشد",
    "هستند", "نیستند", "بودند", "نبودند", "باشند", "نباشند",
    "هستم", "نیستم", "بودم", "نبودم", "باشم", "نباشم",
    "هستی", "نیستی", "بودی", "نبودی", "باشی", "نباشی",
    "هستیم", "نیستیم", "بودیم", "نبودیم", "باشیم", "نباشیم",
    "هستید", "نیستید", "بودید", "نبودید", "باشید", "نباشید",
    # شدن family
    "شد", "نشد", "شده", "نشده",
    "شدم", "شدی", "شدیم", "شدید", "شدند",
    "می‌شود", "نمی‌شود", "می‌شوند", "نمی‌شوند",
    "می‌شد", "نمی‌شد", "می‌شدند", "نمی‌شدند",
    "بشود", "بشوند",
    # کردن family
    "کرد", "نکرد", "کرده", "نکرده",
    "کردم", "کردی", "کردیم", "کردید", "کردند",
    "می‌کند", "نمی‌کند", "می‌کنند", "نمی‌کنند",
    "می‌کرد", "نمی‌کرد", "می‌کردند", "نمی‌کردند",
    "بکند", "بکنند",
    # دادن family
    "داد", "نداد", "داده", "نداده",
    "دادم", "دادی", "دادیم", "دادید", "دادند",
    "می‌دهد", "نمی‌دهد", "می‌دهند", "نمی‌دهند",
    "می‌داد", "نمی‌داد", "بدهد", "بدهند",
    # گرفتن family
    "گرفت", "نگرفت", "گرفته", "نگرفته",
    "گرفتم", "گرفتی", "گرفتیم", "گرفتید", "گرفتند",
    "می‌گیرد", "نمی‌گیرد", "می‌گیرند", "نمی‌گیرند",
    "می‌گرفت", "بگیرد", "بگیرند",
    # رفتن family
    "رفت", "نرفت", "رفته", "نرفته",
    "رفتم", "رفتی", "رفتیم", "رفتید", "رفتند",
    "می‌رود", "نمی‌رود", "می‌روند", "نمی‌روند",
    "می‌رفت", "برود", "بروند",
    # آمدن family
    "آمد", "نیامد", "آمده", "نیامده",
    "آمدم", "آمدی", "آمدیم", "آمدید", "آمدند",
    "می‌آید", "نمی‌آید", "می‌آیند", "نمی‌آیند",
    "می‌آمد", "بیاید", "بیایند",
    # گفتن family
    "گفت", "نگفت", "گفته", "نگفته",
    "گفتم", "گفتی", "گفتیم", "گفتید", "گفتند",
    "می‌گوید", "نمی‌گوید", "می‌گویند",
    "می‌گفت", "بگوید", "بگویند",
    # دیدن family
    "دید", "ندید", "دیده", "ندیده",
    "دیدم", "دیدی", "دیدیم", "دیدند",
    "می‌بیند", "نمی‌بیند", "می‌بینند",
    "می‌دید", "ببیند", "ببینند",
    # خوردن family
    "خورد", "نخورد", "خورده", "نخورده",
    "خوردم", "خوردی", "خوردند",
    "می‌خورد", "نمی‌خورد", "می‌خورند",
    "بخورد", "بخورند",
    # ماندن family
    "ماند", "نماند", "مانده", "نمانده",
    "ماندم", "ماندند",
    "می‌ماند", "نمی‌ماند", "بماند", "بمانند",
    # خواستن family
    "خواست", "نخواست", "خواسته", "نخواسته",
    "خواستم", "خواستند",
    "می‌خواهد", "نمی‌خواهد", "می‌خواهند",
    "بخواهد", "بخواهند",
    # توانستن family
    "توانست", "نتوانست", "توانسته", "نتوانسته",
    "می‌تواند", "نمی‌تواند", "می‌توانند", "نمی‌توانند",
    "بتواند", "بتوانند",
    # دیگر افعال پرکاربرد
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

# All verb phrases, sorted longest-first (word count desc)
_VERB_PHRASES_SORTED = sorted(_VERB_PHRASES, key=lambda s: len(s.split()), reverse=True)

# Function words that should NOT start a new chunk
_NO_SPLIT_BEFORE = {
    "را", "به", "از", "با", "در", "بر", "برای", "بدون",
    "توسط", "نزد", "پیش", "روی", "زیر", "بالای", "کنار", "بین", "میان",
}


# ============================================================
#  Audio helper
# ============================================================

def to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] audio to int16 PCM."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


# ============================================================
#  Persian text normalization
# ============================================================

_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u0640]')

_REMOVE = re.compile(
    r'[«»\u201C\u201D\u2018\u2019`´‹›\[\]\(\)\{\}<>|/\\*#@&^~_=+™©®°•·…\u2013\u2014]+'
)

_CHAR_MAP = str.maketrans({
    'ي': 'ی', 'ك': 'ک', 'ة': 'ه', 'ۀ': 'ه',
    'ؤ': 'و', 'ئ': 'ی', 'أ': 'ا', 'إ': 'ا', 'ٱ': 'ا',
    '٠': '۰', '١': '۱', '٢': '۲', '٣': '۳', '٤': '۴',
    '٥': '۵', '٦': '۶', '٧': '۷', '٨': '۸', '٩': '۹',
    '٫': '.', '٬': ',',
})


def normalize_persian_text(text: str) -> str:
    if not text:
        return ""
    text = text.translate(_CHAR_MAP)
    text = _DIACRITICS.sub('', text)
    text = _REMOVE.sub(' ', text)
    text = re.sub(r'[ \t\u00A0]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================
#  Persian number -> words
# ============================================================

_ONES = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده",
          "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد",
             "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]


def _three_digit_to_words(n: int) -> str:
    parts = []
    h, rest = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if rest:
        if rest < 10:
            parts.append(_ONES[rest])
        elif rest < 20:
            parts.append(_TEENS[rest - 10])
        else:
            t, o = divmod(rest, 10)
            s = _TENS[t]
            if o:
                s += " و " + _ONES[o]
            parts.append(s)
    return " و ".join(parts)


def int_to_persian_words(n: int) -> str:
    if n == 0:
        return _ONES[0]
    if n < 0:
        return "منفی " + int_to_persian_words(-n)
    groups = []
    i = 0
    while n > 0:
        group = n % 1000
        if group:
            words = _three_digit_to_words(group)
            if _SCALES[i]:
                words += " " + _SCALES[i]
            groups.append(words)
        n //= 1000
        i += 1
    return " و ".join(reversed(groups))


_NUM_PATTERN = re.compile(r'[0-9۰-۹]+(?:[.,٫][0-9۰-۹]+)?')
_DIGIT_TRANS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')


def _num_repl(m: re.Match) -> str:
    s = m.group().translate(_DIGIT_TRANS)
    s = s.replace('٫', '.')
    if ',' in s and '.' not in s:
        parts = s.split(',')
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = ''.join(parts)
        else:
            s = s.replace(',', '.')
    if '.' in s:
        int_part, dec_part = s.split('.', 1)
        int_words = int_to_persian_words(int(int_part)) if int_part else _ONES[0]
        dec_words = " ".join(_ONES[int(d)] for d in dec_part if d.isdigit())
        return f"{int_words} ممیز {dec_words}".strip()
    return int_to_persian_words(int(s))


def digits_to_words(text: str) -> str:
    return _NUM_PATTERN.sub(_num_repl, text)


# ============================================================
#  Sentence splitting (Tier 1)
# ============================================================

def split_persian_sentences(text: str):
    text = normalize_persian_text(text)
    text = digits_to_words(text)
    text = re.sub(r'([.?!؛،])(\S)', r'\1 \2', text)
    sentences = re.split(r'(?<=[.?!؛،])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
#  Conjunction splitter (Tier 2)
# ============================================================

def _find_conjunction_indices(words):
    indices = []
    i = 0
    while i < len(words):
        for conj in _CONJ_SORTED:
            conj_words = conj.split()
            n = len(conj_words)
            if i + n <= len(words):
                if all(words[i + j] == conj_words[j] for j in range(n)):
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
    prev_idx = 0
    for idx in conj_indices:
        if idx > prev_idx:
            segment = " ".join(words[prev_idx:idx]).strip()
            if segment:
                segments.append(segment)
        prev_idx = idx
    tail = " ".join(words[prev_idx:]).strip()
    if tail:
        segments.append(tail)

    if len(segments) <= 1:
        return [sentence]

    chunks = []
    current = segments[0]
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


# ============================================================
#  Verb splitter (Tier 3 — SOV clause ends)
# ============================================================

def _find_verb_end_indices(words):
    """Return the index of the LAST word of each verb form found."""
    end_indices = []
    i = 0
    while i < len(words):
        matched_len = 0
        # Multi-word phrases first (longest match)
        for phrase in _VERB_PHRASES_SORTED:
            phrase_words = phrase.split()
            n = len(phrase_words)
            if i + n <= len(words):
                if all(words[i + j] == phrase_words[j] for j in range(n)):
                    matched_len = n
                    break
        # Then single-word verbs
        if matched_len == 0 and words[i] in _VERB_WORDS:
            matched_len = 1

        if matched_len > 0:
            end_indices.append(i + matched_len - 1)
            i += matched_len
        else:
            i += 1
    return end_indices


def split_sentence_at_verbs(sentence: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    """
    Tier-3 fallback. Splits AFTER a verb form (Persian is SOV, so the verb
    usually closes a clause). Skips split points where the next word is a
    function word like را / به / از (those should not start a chunk).
    """
    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    verb_end_indices = _find_verb_end_indices(words)

    if len(verb_end_indices) < MIN_VERB_SPLITS:
        return [sentence]

    split_points = []
    for idx in verb_end_indices:
        if idx + 1 < len(words):
            next_word = words[idx + 1]
            if next_word in _NO_SPLIT_BEFORE:
                continue
        split_points.append(idx)

    if not split_points:
        return [sentence]

    segments = []
    prev = 0
    for idx in split_points:
        segment = " ".join(words[prev:idx + 1]).strip()
        if segment:
            segments.append(segment)
        prev = idx + 1
    tail = " ".join(words[prev:]).strip()
    if tail:
        segments.append(tail)

    if len(segments) <= 1:
        return [sentence]

    chunks = []
    current = segments[0]
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


# ============================================================
#  Chunk builder — tiers in order
# ============================================================

def build_chunks(sentences):
    chunks = []
    for sentence in sentences:
        # Tier 1: sentence already short enough
        if len(sentence) <= MAX_CHARS_PER_CHUNK:
            chunks.append(sentence)
            continue

        # Tier 2: split at conjunctions
        conj_chunks = split_sentence_at_conjunctions(sentence)
        if len(conj_chunks) > 1:
            chunks.extend(conj_chunks)
            continue

        # Tier 3: split after verbs
        verb_chunks = split_sentence_at_verbs(sentence)
        chunks.extend(verb_chunks)

    return chunks


# ============================================================
#  Streaming synthesis
# ============================================================

def synthesize_streaming(text):
    if not text or not text.strip():
        yield None
        return

    sample_rate = model.sample_rate
    yield_every = int(sample_rate * YIELD_INTERVAL_SEC)

    sentences = split_persian_sentences(text)
    chunks = build_chunks(sentences)

    pending_audio = np.zeros(0, dtype=np.float32)
    samples_since_yield = 0

    for i, chunk_text in enumerate(chunks):
        print(f"Streaming chunk {i+1}/{len(chunks)}: {chunk_text[:60]}...")

        try:
            stream = model.generate_audio_stream(
                voice_state, chunk_text, frames_after_eos=0
            )
        except TypeError:
            stream = model.generate_audio_stream(voice_state, chunk_text)

        for frame in stream:
            frame_np = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
            frame_np = frame_np.astype(np.float32).reshape(-1)

            pending_audio = np.concatenate([pending_audio, frame_np])
            samples_since_yield += len(frame_np)

            if samples_since_yield >= yield_every:
                yield (sample_rate, to_int16(pending_audio))
                pending_audio = np.zeros(0, dtype=np.float32)
                samples_since_yield = 0

        silence = np.zeros(int(sample_rate * 0.15), dtype=np.float32)
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

with gr.Blocks(title="Pocket TTS - Farsi (Streaming)") as iface:
    gr.Markdown(
        "## Pocket TTS - Farsi (Persian) — Streaming\n"
        "Paste Persian text and press **Generate**. Audio starts playing "
        "as soon as the first chunk is ready. Press **Stop** to cancel "
        "mid-generation.\n\n"
        "*Long sentences are split in three tiers: (1) sentence boundaries, "
        "(2) Persian conjunctions, (3) Persian verbs (SOV clause ends).*"
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

    gen_event = btn.click(
        fn=synthesize_streaming,
        inputs=txt,
        outputs=out_audio,
    )
    stop_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[gen_event],
    )

iface.queue().launch(server_name="0.0.0.0", server_port=7860)
