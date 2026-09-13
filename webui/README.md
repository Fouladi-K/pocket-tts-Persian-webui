# Pocket TTS - Persian (Farsi) Streaming Web UI

A Gradio-based web interface for the Persian (Farsi) Pocket TTS model.
Paste Persian text, press **Generate**, and hear the audio stream back
in real time — with a **Stop** button to cancel long generations.

---

## Features

- **Streaming playback** — audio starts within ~1 second, regardless of text length
- **Stop button** — cancels generation mid-stream
- **Persian text normalization** — strips guillemets (`«»`), quotes, brackets, diacritics, tatweel, and unifies Arabic letter variants (`ي`→`ی`, `ك`→`ک`, `ة`→`ه`)
- **Digit conversion** — Persian and English digits are converted to Persian words (`۱۲۳` → «صد و بیست و سه», `2024` → «دو هزار و بیست و چهار»), including decimals
- **Three-tier chunking** for long text:
  1. Sentence boundaries (punctuation)
  2. Persian conjunctions (`و`, `اما`, `ولی`, `زیرا`, `چون`, `اگر`, …)
  3. Persian verbs (`است`, `شد`, `می‌شود`, `کرد`, `رفت`, …) — using the SOV clause-end pattern
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

Note: The --extra-index-url flag points pip to the CPU-only PyTorch
wheels. Without it, pip will download the ~3 GB CUDA build even though
Pocket TTS runs fine on CPU. If you already have a GPU and want to use
it, drop the flag.

# 5. Download the voice prompt
wget https://huggingface.co/mehdi-hf/pocket-tts-farsi/resolve/main/example_voice.wav

# Usage
python farsi_webui.py
