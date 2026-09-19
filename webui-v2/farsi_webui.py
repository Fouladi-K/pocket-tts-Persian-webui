import gradio as gr
from pocket_tts import TTSModel
import numpy as np
import re
import statistics
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from normalize_fa import normalize_for_model


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
#  Load TTS model — v2
# ============================================================
model = TTSModel.load_model(
    config="hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml",
    temp=0.3,
)
voice_state = model.get_state_for_audio_prompt("example_voice.wav")


# ============================================================
#  Load G2P model
# ============================================================
G2P_REPO = "mehdi-hf/Homo-GE2PE-Persian-HF"
g2p_tok = AutoTokenizer.from_pretrained(G2P_REPO)
g2p_model = T5ForConditionalGeneration.from_pretrained(G2P_REPO).eval()

TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

YIELD_INTERVAL_SEC = 0.5


# ============================================================
#  Defaults (also exposed in the UI)
# ============================================================
DEFAULT_MAX_CHARS       = 100
DEFAULT_MIN_CHARS       = 30
DEFAULT_SPLIT_COMMA     = True
DEFAULT_RESCUE          = True
DEFAULT_FAE             = 2
DEFAULT_EOS_THRESHOLD   = -4.0

MIN_CONJ_SPLITS = 1
MIN_VERB_SPLITS = 1


# ============================================================
#  Conjunctions
# ============================================================
_CONJUNCTIONS = [
    "به شرط آنکه", "به‌شرط آنکه", "از آنجا که", "ازآنجا که",
    "از این رو", "ازاین‌رو", "با این حال", "بااین‌حال",
    "با اینکه", "بااینکه", "همین که", "همینکه",
    "زیرا که", "زیراکه", "چون که", "چونکه",
    "اگر چه", "اگرچه", "چنان که", "چنانکه",
    "چنان چه", "چنانچه", "بدان که", "بدانکه",
    "و", "یا", "پس", "اگر", "نه", "چون", "اما",
    "خواه", "زیرا", "لیکن", "ولی", "بلکه",
]
_CONJ_SORTED = sorted(_CONJUNCTIONS, key=lambda s: len(s.split()), reverse=True)


# ============================================================
#  Verbs
# ============================================================
_VERB_PHRASES = [
    "شده است", "شده بود", "شده‌اند", "شده بودند", "نشده است", "نشده بود",
    "کرده است", "کرده بود", "کرده‌اند", "کرده بودند", "نکرده است", "نکرده بود",
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
    "رسید","نرسید","رسیده","می‌رسد","برسد","افتاد","افتاده","می‌افتد","بیفتد",
    "نشست","نشسته","می‌نشیند","بنشیند","ایستاد","ایستاده","می‌ایستد","بایستد",
    "برگشت","برگشته","برمی‌گردد","برگردد","مرد","مرده","می‌میرد","بمیرد",
    "خرید","خریده","می‌خرد","بخرد","فروخت","فروخته","می‌فروشد","بفروشد",
    "نوشت","نوشته","می‌نویسد","بنویسد","خواند","خوانده","می‌خواند","بخواند",
    "شنید","شنیده","می‌شنود","بشنود","دانست","دانسته","می‌داند","بداند",
    "فهمید","فهمیده","می‌فهمد","بفهمد",
}
_VERB_PHRASES_SORTED = sorted(_VERB_PHRASES, key=lambda s: len(s.split()), reverse=True)
_NO_SPLIT_BEFORE = {"را","به","از","با","در","بر","برای","بدون",
                    "توسط","نزد","پیش","روی","زیر","بالای","کنار","بین","میان"}

_WEAK_STARTERS = {"و", "یا", "پس", "اگر", "نه", "چون", "اما",
                  "خواه", "زیرا", "لیکن", "ولی", "بلکه"}


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
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


# ============================================================
#  Chunking helpers
# ============================================================
def _hard_split_by_words(text, max_chars):
    """Last-resort word-boundary splitter."""
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


def _pull_weak_starters_back(chunks):
    merged = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        first = c.split(maxsplit=1)[0] if c else ""
        if merged and first in _WEAK_STARTERS:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged


# ============================================================
#  Tier 1 — sentence splitting
# ============================================================
def split_persian_sentences(text, split_on_comma=True):
    text = normalize_for_model(text)
    if split_on_comma:
        text = re.sub(r'([.?!؛؟،])(\S)', r'\1 \2', text)
        sents = re.split(r'(?<=[.?!؛؟،])\s+', text)
    else:
        text = re.sub(r'([.?!؛؟])(\S)', r'\1 \2', text)
        sents = re.split(r'(?<=[.?!؛؟])\s+', text)
    return [s.strip() for s in sents if s.strip()]


# ============================================================
#  Tier 3 — conjunction splitter (attaches conjunctions backward)
# ============================================================
def _find_conj_spans(words):
    spans, i = [], 0
    while i < len(words):
        for c in _CONJ_SORTED:
            cw = c.split(); n = len(cw)
            if i + n <= len(words) and all(words[i + j] == cw[j] for j in range(n)):
                spans.append((i, i + n))
                i += n - 1
                break
        i += 1
    return spans


def split_sentence_at_conjunctions(sentence, max_chars):
    if len(sentence) <= max_chars:
        return [sentence]
    words = sentence.split()
    spans = _find_conj_spans(words)
    if len(spans) < MIN_CONJ_SPLITS:
        return [sentence]
    segs, prev = [], 0
    for start, end in spans:
        if start < prev:
            continue
        seg = " ".join(words[prev:end]).strip()
        if seg:
            segs.append(seg)
        prev = end
    if prev < len(words):
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
#  Chunk builder — tiers + merge + hard split
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

    # merge → weak-starters → hard split → merge → weak-starters
    chunks = _merge_short(chunks, min_chars)
    chunks = _pull_weak_starters_back(chunks)

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

    chunks = _merge_short(chunks, min_chars)
    chunks = _pull_weak_starters_back(chunks)

    return [c for c in chunks if c.strip()]


# ============================================================
#  Streaming synthesis
# ============================================================
def synthesize_streaming(text, max_chars, min_chars, split_on_comma,
                         rescue, frames_after_eos, eos_threshold):
    if not text or not text.strip():
        yield None
        return

    # apply eos_threshold on the model instance (read per-step)
    try:
        model.eos_threshold = float(eos_threshold)
        print(f"→ eos_threshold set to {model.eos_threshold}")
    except Exception as e:
        print(f"→ could not set eos_threshold: {e}")

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

    def generate_for(phonemes, fae):
        try:
            return model.generate_audio_stream(
                voice_state, phonemes, frames_after_eos=fae
            )
        except TypeError:
            return model.generate_audio_stream(voice_state, phonemes)

    for i, raw_text in enumerate(chunks):
        if not raw_text.strip():
            continue

        if buffer_text:
            full_text = buffer_text + " " + raw_text
            buffer_text = None
            label = f"{i+1}/{len(chunks)} [+rescued]"
        else:
            full_text = raw_text
            label = f"{i+1}/{len(chunks)}"

        # G2P (synchronous — needed for rescue correctness)
        try:
            phonemes = phonemise(full_text)
        except Exception as e:
            print(f"  ! G2P failed on chunk {label}: {e}")
            if rescue and i + 1 < len(chunks):
                buffer_text = full_text
            continue

        if not phonemes.strip():
            print(f"  ! chunk {label} produced empty phonemes")
            if rescue and i + 1 < len(chunks):
                buffer_text = full_text
            continue

        print(f"Streaming chunk {label} ({len(full_text)} chars, "
              f"{len(phonemes)} phonemes): {phonemes[:60]}...")

        produced_samples = 0
        try:
            for frame in generate_for(phonemes, frames_after_eos):
                frame_np = (frame.numpy() if hasattr(frame, "numpy")
                            else np.asarray(frame))
                frame_np = frame_np.astype(np.float32).reshape(-1)
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
                buffer_text = full_text
            continue

        if produced_samples == 0:
            print(f"  ! chunk {label} produced 0 samples")

            if rescue and i + 1 < len(chunks):
                print(f"  → rescuing: will retry merged with next chunk")
                buffer_text = full_text
                continue

            if rescue:
                # Last chunk — retry alone with a wider eos_threshold
                print(f"  → last chunk: retrying with eos_threshold -= 1.5")
                orig_eos = getattr(model, "eos_threshold", None)
                try:
                    if orig_eos is not None:
                        model.eos_threshold = orig_eos - 1.5
                    for frame in generate_for(phonemes, frames_after_eos):
                        frame_np = (frame.numpy() if hasattr(frame, "numpy")
                                    else np.asarray(frame))
                        frame_np = frame_np.astype(np.float32).reshape(-1)
                        produced_samples += len(frame_np)
                        pending_audio = np.concatenate([pending_audio, frame_np])
                        samples_since_yield += len(frame_np)
                        out = emit()
                        if out is not None:
                            yield out
                except Exception as e:
                    print(f"    retry failed: {e}")
                finally:
                    if orig_eos is not None:
                        try:
                            model.eos_threshold = orig_eos
                        except Exception:
                            pass

                if produced_samples == 0:
                    print(f"  ! last chunk still empty after retry; giving up")
                    continue

        # Inter-chunk pause only if we produced audio
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
        pending_audio = np.concatenate([pending_audio, silence])
        samples_since_yield += len(silence)
        out = emit(force=True)
        if out is not None:
            yield out

    # Leftover rescued buffer
    if buffer_text and buffer_text.strip():
        print(f"Streaming final rescued chunk: {buffer_text[:60]}...")
        try:
            phonemes = phonemise(buffer_text)
            if phonemes.strip():
                for frame in generate_for(phonemes, frames_after_eos):
                    frame_np = (frame.numpy() if hasattr(frame, "numpy")
                                else np.asarray(frame))
                    frame_np = frame_np.astype(np.float32).reshape(-1)
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
with gr.Blocks(title="Pocket TTS - Farsi v2 (Chunked)") as iface:
    gr.Markdown(
        "## Pocket TTS - Farsi v2 (Persian) — Chunked Streaming\n"
        "Paste Persian text and press **Generate**. Text is split into "
        "chunks, phonemised, then synthesised. Press **Stop** to cancel."
    )
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(
                label="Persian Text",
                lines=10,
                placeholder="متن طولانی خود را اینجا وارد کنید..."
            )
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
                    info="Recommended 30. Merges tiny fragments into "
                         "their neighbor before phonemisation.")
            with gr.Accordion("Model / rescue options", open=False):
                opt_rescue = gr.Checkbox(
                    label="Rescue silent chunks (retry merged with next chunk)",
                    value=DEFAULT_RESCUE)
                opt_fae = gr.Slider(
                    label="frames_after_eos",
                    minimum=0, maximum=16, step=1,
                    value=DEFAULT_FAE,
                    info="Latent frames to allow after EOS. Try 2–4 if "
                         "short chunks come back empty.")
                opt_eos = gr.Slider(
                    label="eos_threshold (less negative = stops sooner)",
                    minimum=-8.0, maximum=-1.0, step=0.5,
                    value=DEFAULT_EOS_THRESHOLD,
                    info="Raise toward -2.0 if generation hits max length "
                         "without EOS. Lower if speech cuts off too early.")
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
        inputs=[txt, opt_max, opt_min, opt_comma, opt_rescue, opt_fae, opt_eos],
        outputs=out_audio,
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

iface.queue().launch(server_name="127.0.0.1", server_port=7862)
