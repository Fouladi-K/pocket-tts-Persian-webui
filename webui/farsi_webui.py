import gradio as gr
from pocket_tts import TTSModel
import numpy as np
import re
import statistics
import pocket_tts.default_parameters as dp


# ============================================================
#  Workaround: statistics.mean([]) raises in tts_model.py
# ============================================================
_orig_mean = statistics.mean
def _safe_mean(data, *args, **kwargs):
    if not data:
        return 0
    return _orig_mean(data, *args, **kwargs)
statistics.mean = _safe_mean


# ============================================================
#  Load TTS model — v1
# ============================================================
dp.MAX_TOKEN_PER_CHUNK = 30

model = TTSModel.load_model(
    config="hf://mehdi-hf/pocket-tts-farsi/farsi.yaml",
    temp=0.3,
)
voice_state = model.get_state_for_audio_prompt("example_voice.wav")

YIELD_INTERVAL_SEC = 0.5


# ============================================================
#  Defaults (also exposed in the UI)
# ============================================================
DEFAULT_MAX_CHARS     = 100
DEFAULT_MIN_CHARS     = 30
DEFAULT_SPLIT_COMMA   = True
DEFAULT_RESCUE        = True
DEFAULT_FAE           = 2
DEFAULT_EOS_THRESHOLD = -4.0

MIN_CONJ_SPLITS = 1
MIN_VERB_SPLITS = 1


# ============================================================
#  Conjunctions
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
#  Verbs (SOV clause-end markers)
# ============================================================
_VERB_PHRASES = [
    "شده است", "شده بود", "شده‌اند", "شده بودند",
    "نشده است", "نشده بود",
    "کرده است", "کرده بود", "کرده‌اند", "کرده بودند",
    "نکرده است", "نکرده بود",
    "رفته است", "رفته بود", "رفته‌اند", "رفته بودند",
    "آمده است", "آمده بود", "آمده‌اند",
    "داده است", "داده بود", "داده‌اند",
    "گرفته است", "گرفته بود", "گفته است", "گفته بود",
    "دیده است", "دیده بود", "خورده است", "خورده بود",
    "مانده است", "مانده بود", "خواسته است", "خواسته بود",
    "توانسته است", "توانسته بود",
]
_VERB_WORDS = {
    "است","هست","نیست","بود","نبود","باشد","نباشد",
    "هستند","نیستند","بودند","نبودند","باشند","نباشند",
    "هستم","نیستم","بودم","نبودم","باشم","نباشم",
    "هستی","نیستی","بودی","نبودی","باشی","نباشی",
    "هستیم","نیستیم","بودیم","نبودیم","باشیم","نباشیم",
    "هستید","نیستید","بودید","نبودید","باشید","نباشید",
    "شد","نشد","شده","نشده","شدم","شدی","شدیم","شدید","شدند",
    "می‌شود","نمی‌شود","می‌شوند","نمی‌شوند","می‌شد","نمی‌شد",
    "می‌شدند","نمی‌شدند","بشود","بشوند",
    "کرد","نکرد","کرده","نکرده","کردم","کردی","کردیم","کردید","کردند",
    "می‌کند","نمی‌کند","می‌کنند","نمی‌کنند","می‌کرد","نمی‌کرد",
    "می‌کردند","نمی‌کردند","بکند","بکنند",
    "داد","نداد","داده","نداده","دادم","دادی","دادیم","دادید","دادند",
    "می‌دهد","نمی‌دهد","می‌دهند","نمی‌دهند","می‌داد","نمی‌داد","بدهد","بدهند",
    "گرفت","نگرفت","گرفته","نگرفته","گرفتم","گرفتی","گرفتیم","گرفتید","گرفتند",
    "می‌گیرد","نمی‌گیرد","می‌گیرند","نمی‌گیرند","می‌گرفت","بگیرد","بگیرند",
    "رفت","نرفت","رفته","نرفته","رفتم","رفتی","رفتیم","رفتید","رفتند",
    "می‌رود","نمی‌رود","می‌روند","نمی‌روند","می‌رفت","برود","بروند",
    "آمد","نیامد","آمده","نیامده","آمدم","آمدی","آمدیم","آمدید","آمدند",
    "می‌آید","نمی‌آید","می‌آیند","نمی‌آیند","می‌آمد","بیاید","بیایند",
    "گفت","نگفت","گفته","نگفته","گفتم","گفتی","گفتیم","گفتید","گفتند",
    "می‌گوید","نمی‌گوید","می‌گویند","می‌گفت","بگوید","بگویند",
    "دید","ندید","دیده","ندیده","دیدم","دیدی","دیدیم","دیدند",
    "می‌بیند","نمی‌بیند","می‌بینند","می‌دید","ببیند","ببینند",
    "خورد","نخورد","خورده","نخورده","خوردم","خوردی","خوردند",
    "می‌خورد","نمی‌خورد","می‌خورند","بخورد","بخورند",
    "ماند","نماند","مانده","نمانده","ماندم","ماندند",
    "می‌ماند","نمی‌ماند","بماند","بمانند",
    "خواست","نخواست","خواسته","نخواسته","خواستم","خواستند",
    "می‌خواهد","نمی‌خواهد","می‌خواهند","بخواهد","بخواهند",
    "توانست","نتوانست","توانسته","نتوانسته",
    "می‌تواند","نمی‌تواند","می‌توانند","نمی‌توانند","بتواند","بتوانند",
    "رسید","نرسید","رسیده","می‌رسد","برسد",
    "افتاد","افتاده","می‌افتد","بیفتد",
    "نشست","نشسته","می‌نشیند","بنشیند",
    "ایستاد","ایستاده","می‌ایستد","بایستد",
    "برگشت","برگشته","برمی‌گردد","برگردد",
    "مرد","مرده","می‌میرد","بمیرد",
    "خرید","خریده","می‌خرد","بخرد",
    "فروخت","فروخته","می‌فروشد","بفروشد",
    "نوشت","نوشته","می‌نویسد","بنویسد",
    "خواند","خوانده","می‌خواند","بخواند",
    "شنید","شنیده","می‌شنود","بشنود",
    "دانست","دانسته","می‌داند","بداند",
    "فهمید","فهمیده","می‌فهمد","بفهمد",
}
_VERB_PHRASES_SORTED = sorted(_VERB_PHRASES, key=lambda s: len(s.split()), reverse=True)

_NO_SPLIT_BEFORE = {
    "را","به","از","با","در","بر","برای","بدون",
    "توسط","نزد","پیش","روی","زیر","بالای","کنار","بین","میان",
}


# ============================================================
#  Audio helper
# ============================================================
def to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


# ============================================================
#  Text normalization
# ============================================================
_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u0640]')
_REMOVE = re.compile(
    r'[«»\u201C\u201D\u2018\u2019`´‹›\[\]\(\)\{\}<>|/\\*#@&^~_=+™©®°•·…\u2013\u2014]+'
)
_CHAR_MAP = str.maketrans({
    'ي':'ی','ك':'ک','ة':'ه','ۀ':'ه','ؤ':'و','ئ':'ی','أ':'ا','إ':'ا','ٱ':'ا',
    '٠':'۰','١':'۱','٢':'۲','٣':'۳','٤':'۴','٥':'۵','٦':'۶','٧':'۷','٨':'۸','٩':'۹',
    '٫':'.','٬':',',
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


# ---- Numbers → Persian words ----
_ONES = ["صفر","یک","دو","سه","چهار","پنج","شش","هفت","هشت","نه"]
_TEENS = ["ده","یازده","دوازده","سیزده","چهارده","پانزده","شانزده","هفده","هجده","نوزده"]
_TENS = ["","","بیست","سی","چهل","پنجاه","شصت","هفتاد","هشتاد","نود"]
_HUNDREDS = ["","صد","دویست","سیصد","چهارصد","پانصد","ششصد","هفتصد","هشتصد","نهصد"]
_SCALES = ["","هزار","میلیون","میلیارد","تریلیون"]


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
    groups, i = [], 0
    while n > 0:
        group = n % 1000
        if group:
            w = _three_digit_to_words(group)
            if _SCALES[i]:
                w += " " + _SCALES[i]
            groups.append(w)
        n //= 1000
        i += 1
    return " و ".join(reversed(groups))


_NUM_PATTERN = re.compile(r'[0-9۰-۹]+(?:[.,٫][0-9۰-۹]+)?')
_DIGIT_TRANS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')


def _num_repl(m: re.Match) -> str:
    s = m.group().translate(_DIGIT_TRANS).replace('٫', '.')
    if ',' in s and '.' not in s:
        parts = s.split(',')
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = ''.join(parts)
        else:
            s = s.replace(',', '.')
    if '.' in s:
        ip, dp = s.split('.', 1)
        iw = int_to_persian_words(int(ip)) if ip else _ONES[0]
        dw = " ".join(_ONES[int(d)] for d in dp if d.isdigit())
        return f"{iw} ممیز {dw}".strip()
    return int_to_persian_words(int(s))


def digits_to_words(text: str) -> str:
    return _NUM_PATTERN.sub(_num_repl, text)


# ============================================================
#  Chunking helpers
# ============================================================
def _hard_split_by_words(text, max_chars):
    words = text.split()
    if not words:
        return [text]
    pieces, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = cur + " " + w
        else:
            pieces.append(cur)
            cur = w
    if cur:
        pieces.append(cur)
    return pieces


def _merge_short(chunks, min_chars):
    if min_chars <= 0:
        return chunks
    merged = []
    for c in chunks:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    if len(merged) >= 2 and len(merged[-1]) < min_chars:
        merged[-2] = (merged[-2] + " " + merged[-1]).strip()
        merged.pop()
    return merged


# ============================================================
#  Tier 1 — sentence splitting
# ============================================================
def split_persian_sentences(text, split_on_comma=True):
    text = normalize_persian_text(text)
    text = digits_to_words(text)
    if split_on_comma:
        text = re.sub(r'([.?!؛،])(\S)', r'\1 \2', text)
        sents = re.split(r'(?<=[.?!؛،])\s+', text)
    else:
        text = re.sub(r'([.?!؛])(\S)', r'\1 \2', text)
        sents = re.split(r'(?<=[.?!؛])\s+', text)
    return [s.strip() for s in sents if s.strip()]


# ============================================================
#  Tier 3 — conjunction splitter
# ============================================================
def _find_conj_indices(words):
    """Return word-index positions where a conjunction starts."""
    idxs, i = [], 0
    while i < len(words):
        for c in _CONJ_SORTED:
            cw = c.split(); n = len(cw)
            if i + n <= len(words) and all(words[i + j] == cw[j] for j in range(n)):
                idxs.append(i)
                i += n - 1
                break
        i += 1
    return idxs


def split_sentence_at_conjunctions(sentence, max_chars):
    """Split long sentences at conjunctions. Conjunctions start the new chunk."""
    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    idxs = _find_conj_indices(words)
    if len(idxs) < MIN_CONJ_SPLITS:
        return [sentence]

    segs, prev = [], 0
    for idx in idxs:
        if idx <= prev:
            continue
        seg = " ".join(words[prev:idx]).strip()
        if seg:
            segs.append(seg)
        prev = idx
    tail = " ".join(words[prev:]).strip()
    if tail:
        segs.append(tail)

    if len(segs) <= 1:
        return [sentence]

    chunks, cur = [], segs[0]
    for s in segs[1:]:
        if len(cur + " " + s) > max_chars and cur.strip():
            chunks.append(cur.strip()); cur = s
        else:
            cur = cur + " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ============================================================
#  Tier 4 — verb splitter
# ============================================================
def _find_verb_end_indices(words):
    ends, i = [], 0
    while i < len(words):
        ml = 0
        for ph in _VERB_PHRASES_SORTED:
            pw = ph.split(); n = len(pw)
            if i + n <= len(words) and all(words[i + j] == pw[j] for j in range(n)):
                ml = n; break
        if ml == 0 and words[i] in _VERB_WORDS:
            ml = 1
        if ml > 0:
            ends.append(i + ml - 1); i += ml
        else:
            i += 1
    return ends


def split_sentence_at_verbs(sentence, max_chars):
    if len(sentence) <= max_chars:
        return [sentence]
    words = sentence.split()
    ve = _find_verb_end_indices(words)
    if len(ve) < MIN_VERB_SPLITS:
        return [sentence]
    sp = []
    for idx in ve:
        if idx + 1 < len(words) and words[idx + 1] in _NO_SPLIT_BEFORE:
            continue
        sp.append(idx)
    if not sp:
        return [sentence]
    segs, prev = [], 0
    for idx in sp:
        s = " ".join(words[prev:idx + 1]).strip()
        if s: segs.append(s)
        prev = idx + 1
    t = " ".join(words[prev:]).strip()
    if t: segs.append(t)
    if len(segs) <= 1:
        return [sentence]
    chunks, cur = [], segs[0]
    for s in segs[1:]:
        if len(cur + " " + s) > max_chars and cur.strip():
            chunks.append(cur.strip()); cur = s
        else:
            cur = cur + " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ============================================================
#  Chunk builder
# ============================================================
def build_chunks(sentences, max_chars, min_chars):
    chunks = []
    for s in sentences:
        if len(s) <= max_chars:
            chunks.append(s); continue
        c = split_sentence_at_conjunctions(s, max_chars)
        if len(c) > 1:
            chunks.extend(c); continue
        chunks.extend(split_sentence_at_verbs(s, max_chars))

    # Merge pass #1
    chunks = _merge_short(chunks, min_chars)

    # Hard split
    enforced, hard_split_count = [], 0
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if len(c) <= max_chars:
            enforced.append(c)
        else:
            pieces = _hard_split_by_words(c, max_chars)
            hard_split_count += 1
            enforced.append(pieces[0])
            enforced.extend(pieces[1:])
    if hard_split_count:
        print(f"→ hard-split {hard_split_count} oversized chunk(s) "
              f"to honour max_chars={max_chars}")
    chunks = enforced

    # Merge pass #2 (post hard-split)
    chunks = _merge_short(chunks, min_chars)

    return [c for c in chunks if c.strip()]


# ============================================================
#  Streaming synthesis
# ============================================================
def _generate_chunk(chunk_text, frames_after_eos):
    try:
        stream = model.generate_audio_stream(
            voice_state, chunk_text, frames_after_eos=frames_after_eos
        )
    except TypeError:
        stream = model.generate_audio_stream(voice_state, chunk_text)
    for frame in stream:
        arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
        yield arr.astype(np.float32).reshape(-1)


def synthesize_streaming(text, max_chars, min_chars, split_on_comma,
                         rescue, frames_after_eos, eos_threshold):
    if not text or not text.strip():
        yield None
        return

    model.eos_threshold = float(eos_threshold)
    print(f"→ eos_threshold set to {model.eos_threshold}")

    sample_rate = model.sample_rate
    yield_every = int(sample_rate * YIELD_INTERVAL_SEC)

    sentences = split_persian_sentences(text, split_on_comma=split_on_comma)
    chunks = build_chunks(sentences, int(max_chars), int(min_chars))

    if not chunks:
        yield None
        return

    print(f"→ {len(chunks)} chunks queued (max={max_chars}, min={min_chars}, "
          f"comma={split_on_comma}, rescue={rescue}, fae={frames_after_eos})")

    pending_audio = np.zeros(0, dtype=np.float32)
    samples_since_yield = 0
    buffer_text = None

    def emit(force=False):
        nonlocal pending_audio, samples_since_yield
        if force or samples_since_yield >= yield_every:
            if len(pending_audio) > 0:
                out = (sample_rate, to_int16(pending_audio))
                pending_audio = np.zeros(0, dtype=np.float32)
                samples_since_yield = 0
                return out
        return None

    for i, raw_chunk in enumerate(chunks):
        if not raw_chunk.strip():
            continue

        if buffer_text:
            chunk_text = buffer_text + " " + raw_chunk
            buffer_text = None
            label = f"{i+1}/{len(chunks)} [+rescued]"
        else:
            chunk_text = raw_chunk
            label = f"{i+1}/{len(chunks)}"

        print(f"Streaming chunk {label} ({len(chunk_text)} chars): "
              f"{chunk_text[:60]}...")

        produced_samples = 0
        try:
            for frame_np in _generate_chunk(chunk_text, frames_after_eos):
                produced_samples += len(frame_np)
                pending_audio = np.concatenate([pending_audio, frame_np])
                samples_since_yield += len(frame_np)
                out = emit()
                if out is not None:
                    yield out
        except Exception as e:
            print(f"  ! chunk {label} raised {type(e).__name__}: {e}")
            if len(pending_audio) > 0:
                yield (sample_rate, to_int16(pending_audio))
                pending_audio = np.zeros(0, dtype=np.float32)
                samples_since_yield = 0
            if rescue and i + 1 < len(chunks):
                print(f"  → rescuing chunk by merging with next")
                buffer_text = chunk_text
            continue

        if produced_samples == 0:
            print(f"  ! chunk {label} produced 0 samples")
            if rescue and i + 1 < len(chunks):
                print(f"  → rescuing: will retry merged with next chunk")
                buffer_text = chunk_text
                continue
            if rescue:
                print(f"  → last chunk: retrying with eos_threshold -= 1.5")
                orig_eos = model.eos_threshold
                try:
                    model.eos_threshold = orig_eos - 1.5
                    for frame_np in _generate_chunk(chunk_text, frames_after_eos):
                        produced_samples += len(frame_np)
                        pending_audio = np.concatenate([pending_audio, frame_np])
                        samples_since_yield += len(frame_np)
                        out = emit()
                        if out is not None:
                            yield out
                except Exception as e:
                    print(f"    retry failed: {e}")
                finally:
                    try:
                        model.eos_threshold = orig_eos
                    except Exception:
                        pass
                if produced_samples == 0:
                    print(f"  ! last chunk still empty after retry; giving up")
                    continue

        silence = np.zeros(int(sample_rate * 0.15), dtype=np.float32)
        pending_audio = np.concatenate([pending_audio, silence])
        samples_since_yield += len(silence)
        out = emit(force=True)
        if out is not None:
            yield out

    if buffer_text and buffer_text.strip():
        print(f"Streaming final rescued chunk: {buffer_text[:60]}...")
        try:
            for frame_np in _generate_chunk(buffer_text, frames_after_eos):
                pending_audio = np.concatenate([pending_audio, frame_np])
                samples_since_yield += len(frame_np)
                out = emit()
                if out is not None:
                    yield out
        except Exception as e:
            print(f"  ! final rescued chunk failed: {e}")

    if len(pending_audio) > 0:
        yield (sample_rate, to_int16(pending_audio))


# ============================================================
#  UI
# ============================================================
with gr.Blocks(title="Pocket TTS - Farsi (Streaming)") as iface:
    gr.Markdown(
        "## Pocket TTS - Farsi (Persian) — Streaming\n"
        "Paste Persian text and press **Generate**."
    )
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Persian Text", lines=10,
                             placeholder="متن طولانی خود را اینجا وارد کنید...")
            with gr.Accordion("Chunking options", open=False):
                opt_comma = gr.Checkbox(
                    label="Split sentences on comma (،)",
                    value=DEFAULT_SPLIT_COMMA)
                opt_max = gr.Slider(
                    label="Max characters per chunk",
                    minimum=20, maximum=500, step=10,
                    value=DEFAULT_MAX_CHARS)
                opt_min = gr.Slider(
                    label="Merge chunks shorter than",
                    minimum=0, maximum=100, step=1,
                    value=DEFAULT_MIN_CHARS,
                    info="Recommended 30. Merges tiny fragments into their "
                         "neighbor before generation.")
            with gr.Accordion("Model / rescue options", open=False):
                opt_rescue = gr.Checkbox(
                    label="Rescue silent chunks (retry merged with next chunk)",
                    value=DEFAULT_RESCUE)
                opt_fae = gr.Slider(
                    label="frames_after_eos",
                    minimum=0, maximum=16, step=1,
                    value=DEFAULT_FAE,
                    info="Latent frames allowed after EOS. Try 2–4 if short "
                         "chunks come back empty.")
                opt_eos = gr.Slider(
                    label="eos_threshold (less negative = stops sooner)",
                    minimum=-8.0, maximum=-1.0, step=0.5,
                    value=DEFAULT_EOS_THRESHOLD,
                    info="Raise toward -2.0 if generation hits max length "
                         "without EOS.")
            with gr.Row():
                btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            clear = gr.ClearButton([txt], value="Clear Text")
        with gr.Column():
            out_audio = gr.Audio(label="Generated Speech",
                                 type="numpy", autoplay=True, streaming=True)

    gen_event = btn.click(
        fn=synthesize_streaming,
        inputs=[txt, opt_max, opt_min, opt_comma,
                opt_rescue, opt_fae, opt_eos],
        outputs=out_audio,
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

iface.queue().launch(server_name="0.0.0.0", server_port=7860)
