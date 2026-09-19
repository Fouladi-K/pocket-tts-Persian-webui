# Pocket TTS - Farsi v2 (Chunked Streaming Web UI)

A Gradio-based web interface for the **v2 Persian Pocket TTS model**.
Paste Persian text, press **Generate**, and hear the audio stream back
in real time — with intelligent multi-tier chunking, silent-chunk
rescue, and live-tunable generation controls to keep long sentences
stable.

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
hard limit ~18), so this script **restores the multi-tier chunking**
from v1 to keep long sentences stable — plus a set of runtime controls
and a rescue path that together eliminate the "missing word" and
"maximum length without EOS" failures that occasionally show up on v2.

---

## Features

- **Streaming playback** — audio starts within ~1–2 seconds
- **Stop button** — cancels generation mid-stream
- **Multi-tier chunking** for long sentences:
  1. Sentence boundaries (`.` `?` `!` `؛` `؟` — plus the Persian comma
     `،` when enabled)
  2. Persian conjunctions (`و`, `اما`, `ولی`, `زیرا`, `چون`, `اگر`, …)
     — attached to the *preceding* segment so no chunk starts with a
     bare conjunction
  3. Persian verbs (`است`, `شد`, `می‌شود`, `کرد`, `رفت`, …) — SOV
     clause-end pattern
  4. Word-boundary fallback — guarantees no chunk ever exceeds the
     configured character limit
- **Short-chunk merging** — fragments below `MIN_CHARS` are absorbed
  into their neighbor, eliminating the model's main failure mode
- **Silent-chunk rescue** — if a chunk raises, produces empty phonemes,
  or yields no audio, it is re-merged with the next chunk and retried
  instead of being silently dropped
- **Live-tunable controls** — chunk size, merge threshold, comma
  splitting, `frames_after_eos`, and `eos_threshold` are all adjustable
  from the UI without restarting
- **Crash guards** — per-chunk `try/except` and a `statistics.mean`
  monkey-patch keep the request alive if the model misbehaves on any
  individual chunk
- **Runs entirely on CPU** — no GPU required

---
## Sample Output

[Listen to a sample output](https://raw.githubusercontent.com/Fouladi-K/pocket-tts-Persian-webui/main/webui-v2/sample_output.wav)

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

All chunking and generation parameters are exposed in the UI under two
accordions. Nothing needs to be edited in code for everyday tuning.

### Chunking options

| Option | Default | Effect |
| :--- | :--- | :--- |
| **Split sentences on comma (،)** | `on` | When on, commas act as sentence boundaries — more, smaller chunks. Turn off for smoother prosody and fewer chunks. |
| **Max characters per chunk** | `100` | Sentences longer than this trigger conjunction and verb splitting. A hard word-boundary fallback guarantees the limit is never exceeded. Lower = smaller chunks, more stable. |
| **Merge chunks shorter than** | `30` | Chunks below this length are absorbed into their neighbor. `0` disables merging. Recommended 30 — short chunks are the model's main failure mode. |

### Model / rescue options

| Option | Default | Effect |
| :--- | :--- | :--- |
| **Rescue silent chunks** | `on` | If a chunk raises, produces empty phonemes, or yields no audio, merge it with the next chunk and retry. The last chunk retries alone with a wider `eos_threshold`. |
| **frames_after_eos** | `2` | Latent frames allowed after EOS. `0` = library default. Try `2`–`4` if short chunks keep coming back empty. |
| **eos_threshold** | `-4.0` | EOS detection threshold. Less negative (e.g. `-2.0`) = model stops sooner; useful when generation hits max length without EOS. More negative (e.g. `-6.0`) = longer output before stopping. |

### Constants still set in code

| Constant | Default | Effect |
| :--- | :--- | :--- |
| `MIN_CONJ_SPLITS` | `1` | Minimum conjunctions required before conjunction splitting fires. |
| `MIN_VERB_SPLITS` | `1` | Minimum verbs required before verb splitting fires. |
| `YIELD_INTERVAL_SEC` | `0.5` | How often audio is pushed to the browser. |

### Recommended tuning

| Symptom | Try |
| :--- | :--- |
| **Runaway generation / garbled audio** | Lower *Max characters per chunk* to `60`–`80`, raise *eos_threshold* toward `-3.0`. |
| **Choppy playback with many small pauses** | Raise *Max characters per chunk* to `150`, keep *Merge chunks shorter than* at `30`. |
| **Maximum generation length reached without EOS** | Raise *eos_threshold* toward `-3.0` or `-2.0`. |
| **Silent chunks still appearing in the log** | Keep *Rescue* on, raise *frames_after_eos* to `3`, and raise *Merge chunks shorter than* to `40`. |
| **Missing words at the end of the paragraph** | *Rescue* should already handle this; verify it is on. If it persists, lower *Max characters per chunk* so fewer tiny fragments are produced. |
| **Prosodic "reset" at chunk boundaries is too audible** | Reduce the inter-chunk silence from `0.25` to `0.10` seconds in `synthesize_streaming`. |

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

### Warnings in the console during generation

- `Maximum generation length reached without EOS` — the model didn't
  find a stopping point for that chunk. It is retried via the rescue
  path. Raise *eos_threshold* if it happens often.
- `hard-split N oversized chunk(s)` — tiers 1–3 didn't decompose a
  chunk and the word-boundary fallback took over. Harmless; raise
  *Max characters per chunk* if you see it frequently.
- `! chunk N/M produced 0 samples` — that chunk yielded no audio. The
  rescue mechanism retries it merged with the next one. Nothing is
  dropped.
- `! chunk N/M raised ...` — an exception inside the model for that
  specific chunk. Also rescued. If it keeps happening, send the
  traceback.

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
