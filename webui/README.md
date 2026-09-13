# Pocket TTS - Farsi (Persian) Streaming Web UI

A Gradio-based web interface for the Persian Pocket TTS model,
with streaming playback, text normalization, and intelligent chunking.

## Based On
- [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) — original TTS engine
- [mallahyari/pocket-tts](https://github.com/mallahyari/pocket-tts) — Persian fork
- [mehdi-hf/pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi) — Persian model

## Features
- Streaming audio playback (starts within ~1 second)
- Persian text normalization (strips guillemets, diacritics, unifies Arabic letters)
- Persian/English digit-to-word conversion
- Three-tier chunking: sentence boundaries → conjunctions → verbs (SOV clause ends)
- Stop button to cancel generation mid-stream

## Installation
pip install pocket-tts gradio scipy numpy --extra-index-url https://download.pytorch.org/whl/cpu

## Usage
python farsi_webui.py
Then open http://localhost:7860

## License
MIT — see LICENSE file.
