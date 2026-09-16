"""Generate audio for the listening set and write a blind A/B scoring page.

The metrics we have cannot see the ezafe -- Persian does not write it, so no
transcription contains it. This is the replacement: a native ear, scoring the
things that actually go wrong, on sentences chosen to provoke them.

    python build_listening_test.py --model v1=hf://mehdi-hf/pocket-tts-farsi/farsi.yaml \
                                   --model v2=hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml \
                                   --voice prompt.wav --out listening

Open listening/index.html, score every pair, then run this with --tally to read
the results back. Model identities are hidden behind A/B labels that are
reshuffled per item, so you cannot drift toward the one you expect to win.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import typer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
SET = Path(__file__).parent / "listening_set.jsonl"

AXES = [
    ("ezafe", "Ezafe", "every link present, none invented"),
    ("onset", "First word", "the opening word is fully there"),
    ("pause", "Phrasing", "pauses fall where the punctuation is"),
    ("natural", "Naturalness", "would pass as a person reading"),
]


def _load():
    return [json.loads(l) for l in SET.read_text(encoding="utf-8").splitlines() if l.strip()]


def _is_phoneme_model(cfg: str) -> bool:
    """v2 takes phonemes; v1 reads Persian script. Guessed from the tokenizer."""
    from pocket_tts.utils.config import load_config

    return "_ph" in str(load_config(cfg).flow_lm.lookup_table.tokenizer_path)


@app.command()
def build(
    model: list[str] = typer.Option(..., help="name=config, repeatable"),
    voice: Path = typer.Option(..., help="voice prompt wav, 5 s or shorter"),
    out: Path = typer.Option(Path("listening")),
    seed: int = typer.Option(0),
    max_tokens: int = typer.Option(18),
) -> None:
    import numpy as np
    import scipy.io.wavfile
    import sphn
    import torch
    from pocket_tts import TTSModel

    from training.farsi.normalize_fa import normalize_for_model
    from training.farsi.synthesize import generate_chunk, split_text, strip_ezafe, trim_silence

    rows = _load()
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)

    # The prompt must obey the 5 s training window, or the model continues it
    # instead of speaking the text -- measured at 0/4 vs 4/4.
    wav, sr = sphn.read(str(voice))
    prompt = out / "voice_prompt.wav"
    sphn.write_wav(str(prompt), wav.mean(axis=0)[: int(5.0 * sr)], int(sr))

    names = []
    for spec in model:
        name, _, cfg = spec.partition("=")
        if not cfg:
            raise typer.BadParameter(f"expected name=config, got {spec!r}")
        names.append(name)
        typer.echo(f"\n=== {name} ===")
        m = TTSModel.load_model(config=cfg, temp=0.3, eos_threshold=-2.0)
        sp = m.flow_lm.conditioner.tokenizer.sp
        phonemic = _is_phoneme_model(cfg)
        g2p = _make_g2p() if phonemic else None
        rate = int(m.mimi.sample_rate)
        state = m.get_state_for_audio_prompt(str(prompt))
        count = (lambda s: len(sp.encode(strip_ezafe(s)))) if phonemic else (lambda s: len(sp.encode(s)))

        for r in rows:
            text = g2p(r["text"]) if phonemic else normalize_for_model(r["text"])
            torch.manual_seed(seed)
            pieces = []
            chunks = split_text(text, count, max_tokens)
            for i, c in enumerate(chunks):
                spoken = strip_ezafe(c) if phonemic else c
                a = generate_chunk(m, state, spoken, frames_after_eos=0, sample_rate=rate)
                pieces.append(trim_silence(np.asarray(a, dtype=np.float32).reshape(-1), rate))
                if i < len(chunks) - 1:
                    pieces.append(np.zeros(int(0.15 * rate), dtype=np.float32))
            joined = np.concatenate(pieces)
            path = out / "audio" / f"{r['id']}__{name}.wav"
            scipy.io.wavfile.write(str(path), rate, (joined * 32767.0).astype(np.int16))
            typer.echo(f"  {r['id']}  {len(joined)/rate:4.1f}s  {r['text'][:40]}")

    # Blind the comparison: A/B order is reshuffled per item and the key is
    # written separately, so a rater cannot drift toward the expected winner.
    if len(names) > 2:
        raise typer.BadParameter("score one model absolutely or two blind; three is not a listening test")
    rng = random.Random(seed)
    key = {}
    for r in rows:
        order = names[:]
        rng.shuffle(order)          # no-op for a single model
        key[r["id"]] = order
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "index.html").write_text(_page(rows, key), encoding="utf-8")
    typer.echo(f"\nopen {out}/index.html — score every pair, then: {Path(__file__).name} tally --out {out}")


def _make_g2p():
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from training.farsi.normalize_fa import normalize_for_model

    G = "mehdi-hf/Homo-GE2PE-Persian-HF"
    tok = AutoTokenizer.from_pretrained(G)
    net = T5ForConditionalGeneration.from_pretrained(G).eval()
    TO = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})
    clause = re.compile(r"(?<=[،؛:])\s+")

    def one(c: str) -> str:
        t = normalize_for_model(c).replace("؟", "").replace("?", "")
        enc = tok([t], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            o = net.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
        return tok.batch_decode(o, skip_special_tokens=True)[0].strip().translate(TO)

    # per clause, matching the shipping path: G2P discards punctuation, so
    # splitting first is the only way a comma survives as a pause
    return lambda text: " ".join(one(c) for c in clause.split(text) if c.strip())


def _page(rows, key) -> str:
    pair = len(next(iter(key.values()))) == 2
    choices = (("A", "A"), ("B", "B"), ("tie", "same")) if pair else (("ok", "ok"), ("bad", "wrong"))
    items = []
    for r in rows:
        models = key[r["id"]]
        axes = "".join(
            f'<div class=ax><span>{label}</span><small>{hint}</small>'
            + "".join(
                f'<label><input type=radio name="{r["id"]}_{ax}" value={v}> {t}</label>'
                for v, t in choices
            )
            + "</div>"
            for ax, label, hint in AXES
        )
        players = "".join(
            f'<b>{"AB"[i] if pair else ""}</b>'
            f'<audio controls preload=none src="audio/{r["id"]}__{m}.wav"></audio>'
            for i, m in enumerate(models)
        )
        items.append(f"""<section>
  <h3>{r['id']} <em>{r['cat']}</em></h3>
  <p class=fa dir=rtl>{r['text']}</p>
  <p class=hint>Listen for: {r['listen']}</p>
  <div class=row>{players}</div>
  {axes}
</section>""")
    return f"""<!doctype html><meta charset=utf-8><title>Listening test</title>
<style>
 body{{font:15px/1.6 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem;color:#111}}
 section{{border:1px solid #e5e7eb;border-radius:10px;padding:1rem;margin:1rem 0}}
 h3{{margin:0 0 .4rem;font-size:15px}} h3 em{{color:#6b7280;font-style:normal;font-weight:400}}
 .fa{{font-size:20px;margin:.3rem 0}} .hint{{color:#6b7280;margin:.2rem 0 .8rem}}
 .row{{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.6rem}}
 .ax{{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;padding:.25rem 0}}
 .ax span{{min-width:92px;font-weight:600}} .ax small{{color:#6b7280;min-width:250px}}
 button{{font:inherit;padding:.6rem 1rem;border-radius:8px;border:1px solid #111;background:#111;color:#fff;cursor:pointer}}
 #out{{white-space:pre-wrap;background:#f8fafc;padding:1rem;border-radius:8px}}
</style>
<h1>Listening test</h1>
<p>{"A and B are two models, hidden and reshuffled per item." if pair else
      "One model, scored on its own terms."} Score each axis, or leave it blank
if the sentence does not test it. Then press save.</p>
{''.join(items)}
<button onclick="save()">Save scores</button>
<pre id=out></pre>
<script>
function save(){{
  const d={{}};
  document.querySelectorAll('input[type=radio]:checked').forEach(i=>d[i.name]=i.value);
  const blob=new Blob([JSON.stringify(d,null,1)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='scores.json';a.click();
  document.getElementById('out').textContent=Object.keys(d).length+' scores saved to scores.json';
}}
</script>"""


@app.command()
def tally(out: Path = typer.Option(Path("listening"))) -> None:
    """Unblind scores.json and report per axis and per category."""
    key = json.loads((out / "key.json").read_text())
    scores = json.loads((out / "scores.json").read_text())
    rows = {r["id"]: r for r in _load()}
    wins: dict = {}
    for name, choice in scores.items():
        rid, _, ax = name.rpartition("_")
        if rid not in key:
            continue
        models = key[rid]
        if len(models) == 2:
            winner = "tie" if choice == "tie" else (models[0] if choice == "A" else models[1])
        else:
            winner = f"{models[0]}:{choice}"
        wins.setdefault(ax, {}).setdefault(winner, 0)
        wins[ax][winner] += 1
        cat = rows[rid]["cat"]
        wins.setdefault(f"cat:{cat}", {}).setdefault(winner, 0)
        wins[f"cat:{cat}"][winner] += 1
    for k in sorted(wins):
        total = sum(wins[k].values())
        parts = "  ".join(f"{n} {c}/{total}" for n, c in sorted(wins[k].items(), key=lambda x: -x[1]))
        typer.echo(f"  {k:22} {parts}")


if __name__ == "__main__":
    app()
