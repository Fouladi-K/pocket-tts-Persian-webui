"""Gradio UI for the ezafe A/B test: is it lost by the G2P, or by the model?

v2 does not read Persian. A separate G2P decides where every ezafe goes and the
model renders what it is handed, so a missing ezafe has two possible causes
calling for opposite work. Phonemise, correct by hand, and hear both.

    MODEL_CFG=scratchpad/v2model/model.yaml .venv/bin/python training/farsi/v2/ezafe_ab_app.py
"""

from __future__ import annotations

import difflib
import os
import re
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import scipy.io.wavfile
import sphn
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pocket_tts import TTSModel  # noqa: E402
from training.farsi.normalize_fa import normalize_for_model  # noqa: E402
from training.farsi.synthesize import (  # noqa: E402
    generate_chunk,
    split_text,
    strip_ezafe,
    trim_silence,
)
from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa: E402

MODEL_CFG = os.environ.get("MODEL_CFG", "hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml")
G2P_ID = "mehdi-hf/Homo-GE2PE-Persian-HF"
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

# The Space splits on clause punctuation BEFORE phonemising, because G2P discards
# it. That changes what G2P sees, and its ezafe decisions are context-dependent:
# on one test sentence the whole-sentence pass produced "CandbarAbariye qeymate"
# (correct) where per-clause dropped the ezafe, while per-clause produced
# "tarke tahsile dAneSAmuzAn" (correct) where whole-sentence added a spurious one.
# Neither context wins outright -- but this tool has to match what the user
# actually hears, or correcting its phonemes answers the wrong question.
CLAUSE_SPLIT = re.compile(r"(?<=[،؛:])\s+")

DEFAULT_VOICE = str(REPO / "training/farsi/v2/cv_eval/audio/common_voice_fa_19227531.wav")
DEFAULT_TEXT = "در آستانه آغاز سال تحصیلی جدید، فشار اقتصادی بر خانواده‌ها تشدید شده است"

model = TTSModel.load_model(config=MODEL_CFG, temp=0.3, eos_threshold=-2.0)
_sp = model.flow_lm.conditioner.tokenizer.sp
_g2p_tok = AutoTokenizer.from_pretrained(G2P_ID)
_g2p = T5ForConditionalGeneration.from_pretrained(G2P_ID).eval()
RATE = int(model.mimi.sample_rate)


def _count(s: str) -> int:
    return len(_sp.encode(strip_ezafe(s)))


def phonemise(persian: str) -> tuple[str, str, str]:
    """Persian -> phonemes. Returns it three times: shown, editable, and kept for the diff."""
    if not persian or not persian.strip():
        raise gr.Error("Enter some Persian text first.")
    def one(clause: str) -> str:
        text = normalize_for_model(clause).replace("؟", "").replace("?", "")
        enc = _g2p_tok([text], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = _g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
        return _g2p_tok.batch_decode(out, skip_special_tokens=True)[0].strip().translate(TO_PHONEMES)

    p = " ".join(one(c) for c in CLAUSE_SPLIT.split(persian) if c.strip())
    return p, p, p


def _prompt_path(voice) -> str:
    """Cap the prompt at 5 s: longer is out of distribution and the model
    continues the prompt instead of speaking the text."""
    if voice is None:
        wav, sr = sphn.read(DEFAULT_VOICE)
        data, sr = wav.mean(axis=0), int(sr)
    else:
        sr, data = voice
        data = np.asarray(data)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            data = data.astype(np.float32) / max(abs(info.min), info.max)
        else:
            data = data.astype(np.float32)
    data = data[: int(5.0 * sr)]
    peak = float(np.max(np.abs(data))) or 1.0
    if peak > 1.0:
        data = data / peak
    path = Path(tempfile.mkdtemp()) / "voice_prompt.wav"
    scipy.io.wavfile.write(str(path), int(sr), (data * 32767.0).astype(np.int16))
    return str(path)


def _speak(state, phonemes: str, max_tokens: int, temperature: float, seed: int):
    model.temp = float(temperature)
    torch.manual_seed(int(seed))
    pieces = []
    chunks = split_text(phonemes, _count, int(max_tokens))
    for i, chunk in enumerate(chunks):
        audio = generate_chunk(model, state, strip_ezafe(chunk),
                               frames_after_eos=0, sample_rate=RATE)
        pieces.append(trim_silence(np.asarray(audio, dtype=np.float32).reshape(-1), RATE))
        if i < len(chunks) - 1:
            pieces.append(np.zeros(int(0.15 * RATE), dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32), len(chunks)


def compare(original: str, corrected: str, voice, max_tokens: int, temperature: float, seed: int):
    """Speak both, everything held equal except the text."""
    original = " ".join((original or "").split())
    corrected = " ".join((corrected or "").split())
    if not original or not corrected:
        raise gr.Error("Phonemise first, then edit the right-hand box.")

    if original == corrected:
        diff = "**Nothing edited yet** — both sides will be identical. Change the phonemes on the right."
    else:
        rows = [ln for ln in difflib.unified_diff(original.split(), corrected.split(), lineterm="", n=0)
                if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
        diff = "**Your edits**\n\n```\n" + "\n".join(rows) + "\n```"

    state = model.get_state_for_audio_prompt(_prompt_path(voice))
    a, na = _speak(state, original, max_tokens, temperature, seed)
    b, nb = _speak(state, corrected, max_tokens, temperature, seed)
    note = (f"\n\nA: {len(a)/RATE:.1f}s, {na} chunk(s) · B: {len(b)/RATE:.1f}s, {nb} chunk(s)"
            "\n\nB sounds right → the **frontend** is the bottleneck, G2P work pays off."
            "\n\nB still drops it → the **model** is, and G2P work will not help.")
    return (RATE, a), (RATE, b), diff + note


with gr.Blocks(title="Ezafe A/B") as demo:
    gr.Markdown(
        "# Ezafe A/B\n"
        "v2 reads phonemes, not Persian, so a separate G2P decides every ezafe and the "
        "model renders what it is given. When an ezafe is missing, that is either the "
        "frontend's fault or the model's — and the two call for opposite work.\n\n"
        "Phonemise, correct the phonemes by hand on the right, and hear both. Same seed, "
        "same prompt, same settings: only the text differs.\n\n"
        "`A` long ā · `a` short a · `?` glottal stop · `S` š · `C` č · `;` ž · `x` خ · `q` ق/غ. "
        "The ezafe is a trailing `e`/`ye` on the **first** word of the pair. A trailing `1` "
        "only tells the chunker not to split the pair; it is stripped before generation, so "
        "it changes phrasing, not sound."
    )
    original = gr.State("")
    with gr.Row():
        persian = gr.Textbox(label="Persian text", value=DEFAULT_TEXT, lines=3,
                             rtl=True, text_align="right")
        voice = gr.Audio(label="Voice prompt (optional, capped at 5 s)", type="numpy",
                         sources=["upload", "microphone"])
    go_ph = gr.Button("Phonemise", variant="secondary")
    with gr.Row():
        shown = gr.Textbox(label="A — what the G2P produced", lines=4, interactive=False)
        edited = gr.Textbox(label="B — your correction (edit this)", lines=4, interactive=True)
    with gr.Accordion("Settings", open=False):
        max_tokens = gr.Slider(8, 24, value=18, step=1, label="Max tokens per chunk")
        temperature = gr.Slider(0.05, 1.0, value=0.3, step=0.05, label="Temperature")
        seed = gr.Slider(0, 50, value=0, step=1, label="Seed (same for both sides)")
    go = gr.Button("Speak both", variant="primary")
    with gr.Row():
        out_a = gr.Audio(label="A — G2P as-is", type="numpy")
        out_b = gr.Audio(label="B — corrected", type="numpy")
    report = gr.Markdown()

    go_ph.click(phonemise, inputs=[persian], outputs=[shown, edited, original])
    go.click(compare, inputs=[original, edited, voice, max_tokens, temperature, seed],
             outputs=[out_a, out_b, report])

if __name__ == "__main__":
    demo.queue().launch()
