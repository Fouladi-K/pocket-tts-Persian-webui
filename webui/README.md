# Pocket TTS - Farsi Streaming Web UI

A Gradio-based web interface for the Persian (Farsi) Pocket TTS model.
Paste Persian text, press **Generate**, and hear the audio stream back in
real time — with a **Stop** button to cancel long generations.

This is a subfolder of
[pocket-tts-Persian-webui](https://github.com/Fouladi-K/pocket-tts-Persian-webui),
which is a fork of
[mallahyari/pocket-tts](https://github.com/mallahyari/pocket-tts).

---

## Features

- **Streaming playback** — audio starts within ~1 second, regardless of text length
- **Stop button** — cancels generation mid-stream
- **Persian text normalization** — strips guillemets (`«»`), quotes, brackets,
  diacritics, tatweel, and unifies Arabic letter variants
  (`ي` → `ی`, `ك` → `ک`, `ة` → `ه`)
- **Digit conversion** — Persian and English digits are converted to Persian
  words (`۱۲۳` → «صد و بیست و سه», `2024` → «دو هزار و بیست و چهار»),
  including decimals
- **Three-tier chunking** for long text:
  1. Sentence boundaries (punctuation)
  2. Persian conjunctions (`و`, `اما`, `ولی`, `زیرا`, `چون`, `اگر`, …)
  3. Persian verbs (`است`, `شد`, `می‌شود`, `کرد`, `رفت`, …) — using the
     SOV clause-end pattern
- **Runs entirely on CPU** — no GPU required

---

## Requirements

- Ubuntu (tested), or any Linux distribution with Python 3.10–3.14
- Python 3.10+ with `venv`
- ~1.5 GB free disk space (for the CPU PyTorch wheel and the model)
- ~1 GB free RAM at runtime

---

## Installation

Open a terminal, navigate to this `webui/` folder, and run:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies (CPU-only PyTorch)
pip install pocket-tts gradio scipy --extra-index-url https://download.pytorch.org/whl/cpu
```

> **Note:** The `--extra-index-url` flag points pip to the CPU-only PyTorch
> wheels. Without it, pip will download the ~3 GB CUDA build even though
> Pocket TTS runs fine on CPU. If you already have a GPU and want to use
> it, drop the flag.

### Download the voice prompt

The Farsi model uses a reference voice to anchor the speaker identity.
Download `example_voice.wav` from the Hugging Face model page and place it
in this folder:

```bash
wget https://huggingface.co/mehdi-hf/pocket-tts-farsi/resolve/main/example_voice.wav
```

If `wget` is not installed, use `curl -O <url>` or install it with
`sudo apt install wget`.

---

## Usage

With the virtual environment active:

```bash
python farsi_webui.py
```

On the **first run**, the script downloads the Farsi model from Hugging
Face (~1–2 GB). This can take a few minutes depending on your internet
speed. The model is cached afterwards, so subsequent runs start instantly.

Once you see:

```
Running on local URL:  http://0.0.0.0:7860
```

Open your browser and go to:

```
http://localhost:7860
```

Paste Persian text into the input box, press **Generate**, and the audio
will start playing as soon as the first chunk is ready. Press **Stop** at
any time to cancel. Press **Clear Text** to reset the input.

---

## Configuration

The chunking behavior is controlled by constants at the top of
`farsi_webui.py`:

| Constant | Default | Effect |
| :--- | :--- | :--- |
| `MAX_CHARS_PER_CHUNK` | `150` | Sentences longer than this trigger the conjunction and verb splitters. Lower = more, smaller chunks (more stable). Higher = fewer, longer chunks (smoother). |
| `MIN_CONJ_SPLITS` | `1` | Minimum number of conjunctions required before the conjunction splitter fires. |
| `MIN_VERB_SPLITS` | `1` | Minimum number of verbs required before the verb splitter fires. |
| `YIELD_INTERVAL_SEC` | `0.5` | How often new audio is pushed to the browser, in seconds. |

**Recommended tuning:**

- **Farsi model is unstable / producing artifacts** → keep
  `MAX_CHARS_PER_CHUNK` at `150` or lower.
- **Audio sounds choppy (too many pauses)** → raise `MAX_CHARS_PER_CHUNK`
  to `250` or `300`.
- **Generation is slow on long texts** → lower `YIELD_INTERVAL_SEC` to
  `0.25` for snappier streaming (at the cost of more network round-trips).

---

## Troubleshooting

### Warning: `Trying to convert audio automatically from float32 to 16-bit int format`

This warning is suppressed by the script because it converts audio to
`int16` before yielding. If you see it, you are running an older version
of the script — update to the latest `farsi_webui.py`.

### CUDA error on startup

The CPU-only install should prevent this. If it happens anyway:

```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Model download fails

Check your internet connection and try again. You can also download the
model manually from
[mehdi-hf/pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi)
and place the files in the Hugging Face cache at `~/.cache/huggingface/hub`.

### Gradio does not open on port 7860

Another process is using the port. Edit the `server_port` argument at the
bottom of `farsi_webui.py` and change it to `7861`, `7862`, etc.

### Long sentences still fail

Lower `MAX_CHARS_PER_CHUNK` to `120` or `100`. If the failure persists,
add the specific verb form that appears in your failing sentence to the
`_VERB_WORDS` set in `farsi_webui.py`.

---

## Files

| File | Purpose |
| :--- | :--- |
| `farsi_webui.py` | Main Gradio application |
| `example_voice.wav` | Reference voice prompt (download separately) |
| `venv/` | Python virtual environment (not committed to git) |
| `README.md` | This file |

---

## Attribution

- [Kyutai Labs — Pocket TTS](https://github.com/kyutai-labs/pocket-tts) — original TTS engine
- [mallahyari/pocket-tts](https://github.com/mallahyari/pocket-tts) — Persian fork
- [mehdi-hf/pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi) — Persian model on Hugging Face

## License

MIT — see the [LICENSE](../LICENSE) file in the repository root.
