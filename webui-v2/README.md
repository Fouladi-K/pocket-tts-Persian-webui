# Pocket TTS - Farsi v2 (Chunked Streaming Web UI)

A Gradio-based web interface for the **v2 Persian Pocket TTS model**.
Paste Persian text, press **Generate**, and hear the audio stream back
in real time — with intelligent three-tier chunking to keep long
sentences stable.

This is a subfolder of
[pocket-tts-Persian-webui](https://github.com/Fouladi-k/pocket-tts-Persian-webui),
which is a fork of
[mallahyari/pocket-tts](https://github.com/mallahyari/pocket-tts).

---

## What's new in v2

The v2 model is **phoneme-based**, not script-based. Passing Persian
text directly produces silence, so this script runs a
**grapheme-to-phoneme (G2P)** step first. The result is:

- More natural prosody
- Better handling of loan words and ezafe
- **Dramatically lower runaway-generation rate** (0–1 per 300 sentences
  vs 25 per 300 in v1)

The trade-off: v2 works on short phoneme sequences (target ~11 tokens,
hard limit ~18), so this script **restores the three-tier chunking**
from v1 to keep long sentences stable.

---

## Features

- **Streaming playback** — audio starts within ~1–2 seconds
- **Stop button** — cancels generation mid-stream
- **Threaded G2P** — phonemisation of the next sentence runs in a
  background thread while the current sentence streams, so
  inter-sentence gaps stay at ~250 ms instead of ~500 ms
- **Three-tier chunking** for long sentences:
  1. Sentence boundaries (punctuation)
  2. Persian conjunctions (`و`, `اما`, `ولی`, `زیرا`, `چون`, `اگر`, …)
  3. Persian verbs (`است`, `شد`, `می‌شود`, `کرد`, `رفت`, …) — SOV clause-end pattern
- **Runs entirely on CPU** — no GPU required

---
## Sample Output

<audio controls>
  <source src="https://raw.githubusercontent.com/Fouladi-K/pocket-tts-Persian-webui/main/webui-v2/sample_output.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

---

## Requirements

- Ubuntu (tested) or any Linux distribution with Python 3.10–3.14
- Python 3.10+ with `venv`
- ~3 GB free disk space (CPU PyTorch + both models)
- ~1.5 GB free RAM at runtime

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install the Farsi TTS fork (CPU-only PyTorch)
pip install "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main" \
  --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Install Gradio, SciPy, and G2P dependencies (CPU-only)
pip install gradio scipy transformers \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

### Verify CPU-only install

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want a version ending in `+cpu` and `False`.

### Download required files

Both must sit in the same folder as `farsi_webui.py`:

```bash
wget https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/normalize_fa.py
wget https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/example_voice.wav
```

`example_voice.wav` must be **≤ 5 seconds**. Longer prompts cause the
model to continue the prompt's speech instead of your text.

---

## Usage

```bash
python farsi_webui.py
```

**First run** downloads two models from Hugging Face:

- `mehdi-hf/pocket-tts-farsi-v2` 
- `mehdi-hf/Homo-GE2PE-Persian-HF` 

Subsequent runs start instantly from cache.

Open your browser at `http://localhost:7862`.


---

## Configuration

Constants at the top of `farsi_webui.py`:

| Constant | Default | Effect |
| :--- | :--- | :--- |
| `MAX_CHARS_PER_CHUNK` | `150` | Sentences longer than this trigger tiers 2 and 3. Lower = smaller chunks, more stable. |
| `MIN_CONJ_SPLITS` | `1` | Minimum conjunctions required before tier 2 fires. |
| `MIN_VERB_SPLITS` | `1` | Minimum verbs required before tier 3 fires. |
| `YIELD_INTERVAL_SEC` | `0.5` | How often audio is pushed to the browser. |

**Recommended tuning:**

- **Runaway generation / garbled audio** → lower `MAX_CHARS_PER_CHUNK` to `120`
- **Choppy playback with many small pauses** → raise `MAX_CHARS_PER_CHUNK` to `200`–`250`
- **Prosodic "reset" at chunk boundaries is too audible** → reduce
  inter-chunk silence from `0.25` to `0.10` seconds

---

## Troubleshooting

### `ValidationError` about `capitalize_first_letter`

You installed `pocket-tts` from PyPI instead of the fork. Re-run:

```bash
pip uninstall pocket-tts -y
pip install "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main"
```

### Silence instead of speech

`normalize_fa.py` is missing or not in the same folder as
`farsi_webui.py`. The G2P step is mandatory for v2.

### Model continues the voice prompt instead of your text

`example_voice.wav` is longer than 5 seconds. Replace it with a shorter
clip.

### `transformers` tied-weights warning on startup

Harmless. The model loads correctly with the saved weights. It prints
once per session.

### CUDA was installed by mistake

```bash
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Files

| File | Purpose |
| :--- | :--- |
| `farsi_webui.py` | Main Gradio application |
| `normalize_fa.py` | Text normalization + digit handling (from HF repo) |
| `example_voice.wav` | Reference voice prompt (≤ 5 s, from HF repo) |
| `requirements.txt` | Dependencies |
| `venv/` | Python virtual environment (not committed) |
| `README.md` | This file |

---

## Attribution

- [Kyutai Labs — Pocket TTS](https://github.com/kyutai-labs/pocket-tts) — original TTS engine
- [mallahyari/pocket-tts](https://github.com/mallahyari/pocket-tts) — Persian fork
- [mehdi-hf/pocket-tts-farsi-v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2) — v2 Persian model
- [mehdi-hf/Homo-GE2PE-Persian-HF](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) — Persian G2P model

## License

MIT — see the [LICENSE](../LICENSE) file in the repository root.
