#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "sphn", "torch", "transformers", "sentencepiece", "typer"]
# ///
"""Is the ezafe lost by the G2P, or by the model?

v2 does not read Persian. A separate grapheme-to-phoneme stage decides where
every ezafe goes, and the model renders whatever it is handed. So when an ezafe
is missing from the audio there are two candidates, and they call for opposite
work: fix the frontend, or fix the model.

This settles it. Phonemise a sentence, correct the phonemes by hand, and listen
to both. If the corrected one is right, the frontend is the bottleneck. If it
drops the ezafe too, no amount of G2P work will help.

    uv run training/farsi/v2/ezafe_ab.py prepare --text "در آستانه آغاز سال تحصیلی جدید"
    $EDITOR ezafe_ab/corrected.txt          # fix the phonemes
    uv run training/farsi/v2/ezafe_ab.py speak --voice prompt.wav

Then listen to ezafe_ab/a_g2p.wav against ezafe_ab/b_corrected.wav.
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

DEFAULT_CONFIG = "hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml"
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


LEGEND = """\
# Edit the phonemes below, then run:  ezafe_ab.py speak --voice <prompt.wav>
#
# Notation: A = long ā   a = short a   ? = glottal stop (ع/ء)   S = š   C = č
#           ; = ž        x = خ         q = ق/غ
#
# The ezafe is a trailing "e" (or "ye" after a vowel) on the FIRST word of the
# pair: "ketAbe man" is ketāb-e man. A trailing "1" marks it for the chunker so
# it never splits the pair across a chunk boundary; the "1" is stripped before
# the model sees the text, so adding or removing one changes phrasing, not sound.
#
# Lines starting with # are ignored.
"""


def _phonemise(persian: str) -> str:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from training.farsi.normalize_fa import normalize_for_model

    tok = AutoTokenizer.from_pretrained(G2P_ID)
    g2p = T5ForConditionalGeneration.from_pretrained(G2P_ID).eval()
    def one(clause: str) -> str:
        text = normalize_for_model(clause).replace("؟", "").replace("?", "")
        enc = tok([text], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip().translate(TO_PHONEMES)

    parts = [one(c) for c in CLAUSE_SPLIT.split(persian) if c.strip()]
    return " ".join(parts)


def _read_phonemes(path: Path) -> str:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    return " ".join(" ".join(lines).split())


@app.command()
def prepare(
    text: str = typer.Option(..., help="Persian text, in Persian script"),
    out: Path = typer.Option(Path("ezafe_ab"), help="working directory"),
) -> None:
    """Phonemise the text and write a copy for you to correct by hand."""
    out.mkdir(parents=True, exist_ok=True)
    phon = _phonemise(text)
    (out / "persian.txt").write_text(text + "\n", encoding="utf-8")
    (out / "g2p.txt").write_text(phon + "\n", encoding="utf-8")
    (out / "corrected.txt").write_text(LEGEND + "\n" + phon + "\n", encoding="utf-8")
    typer.echo(f"\n{phon}\n")
    typer.echo(f"wrote {out}/corrected.txt — fix the phonemes there, then:")
    typer.echo(f"  uv run {Path(__file__).name} speak --voice <prompt.wav> --out {out}")


@app.command()
def speak(
    voice: Path = typer.Option(..., help="voice prompt wav, 5 s or shorter"),
    out: Path = typer.Option(Path("ezafe_ab"), help="working directory from `prepare`"),
    config: str = typer.Option(DEFAULT_CONFIG, help="model config"),
    max_tokens: int = typer.Option(18),
    temperature: float = typer.Option(0.3),
    seed: int = typer.Option(0, help="same seed for both, so only the text differs"),
) -> None:
    """Speak the G2P phonemes and your corrected ones, with everything else held equal."""
    import numpy as np
    import scipy.io.wavfile
    import torch
    from pocket_tts import TTSModel

    from training.farsi.synthesize import generate_chunk, split_text, strip_ezafe, trim_silence

    g2p_text = _read_phonemes(out / "g2p.txt")
    fixed = _read_phonemes(out / "corrected.txt")
    if not fixed:
        raise typer.BadParameter(f"{out}/corrected.txt is empty — run `prepare` first")

    if g2p_text == fixed:
        typer.secho("corrected.txt is unchanged — edit it first, or this compares "
                    "the same text twice", fg="yellow")
    else:
        typer.echo("\nwhat you changed:")
        for line in difflib.unified_diff(g2p_text.split(), fixed.split(), lineterm="", n=1):
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                typer.echo(f"  {line}")

    model = TTSModel.load_model(config=config, temp=temperature, eos_threshold=-2.0)
    sp = model.flow_lm.conditioner.tokenizer.sp
    count = lambda s: len(sp.encode(strip_ezafe(s)))  # noqa: E731

    # The prompt has to obey the 5 s training cap; longer and the model tends to
    # continue the prompt instead of speaking the text.
    import sphn

    wav, sr = sphn.read(str(voice))
    trimmed = out / "voice_prompt.wav"
    sphn.write_wav(str(trimmed), wav.mean(axis=0)[: int(5.0 * sr)], int(sr))
    state = model.get_state_for_audio_prompt(str(trimmed))
    rate = int(model.mimi.sample_rate)

    for name, text in (("a_g2p", g2p_text), ("b_corrected", fixed)):
        torch.manual_seed(seed)
        pieces = []
        chunks = split_text(text, count, max_tokens)
        for i, chunk in enumerate(chunks):
            audio = generate_chunk(model, state, strip_ezafe(chunk),
                                   frames_after_eos=0, sample_rate=rate)
            pieces.append(trim_silence(np.asarray(audio, dtype=np.float32).reshape(-1), rate))
            if i < len(chunks) - 1:
                pieces.append(np.zeros(int(0.15 * rate), dtype=np.float32))
        joined = np.concatenate(pieces)
        path = out / f"{name}.wav"
        scipy.io.wavfile.write(str(path), rate, (joined * 32767.0).astype(np.int16))
        typer.echo(f"  {path}  {len(joined) / rate:.1f}s  ({len(chunks)} chunk(s))")

    typer.echo(f"\nListen to {out}/a_g2p.wav against {out}/b_corrected.wav.")
    typer.echo("Corrected sounds right -> the frontend is the bottleneck, G2P work pays off.")
    typer.echo("Corrected still drops it -> the model is the bottleneck, G2P work will not help.")


if __name__ == "__main__":
    app()
