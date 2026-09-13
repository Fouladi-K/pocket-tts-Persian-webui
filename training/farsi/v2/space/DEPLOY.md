# Deploying the v2 demo Space

Everything here is tested end to end on CPU (4.2x faster than real time,
including the G2P pass). It is not deployed: creating a Gradio Space requires a
PRO subscription on a personal account, which is why Hugging Face hosted the v1
demo under `hugging-apps` rather than under `mehdi-hf`.

To deploy, from this directory:

```python
from huggingface_hub import HfApi
api = HfApi()
repo = "<owner>/pocket-tts-farsi-v2-demo"      # a PRO account, or an org with Space quota
api.create_repo(repo_id=repo, repo_type="space", space_sdk="gradio")
api.upload_folder(folder_path=".", repo_id=repo, repo_type="space")
```

Any example voice works now that the app caps it, but prefer a clip already
under 5 seconds. Training capped voice prompts at 5 s, and the app used to hand
its default through untrimmed: a 6.1 s clip scored 0/4 on a first-word test
whole and 4/4 cut to 5 s.

`normalize_fa.py` and `example_voice.wav` are not kept here — they live in
`training/farsi/` and `training/farsi/v2/cv_eval/audio/` respectively. Copy both
into the upload folder first:

```bash
cp ../../normalize_fa.py .
cp ../cv_eval/audio/common_voice_fa_19227531.wav example_voice.wav
```

## Why requirements.txt installs pocket-tts from git

The released package rejects this model's `model.yaml`: it does not know
`capitalize_first_letter`, and the config forbids unknown keys. Without that
flag the text frontend upper-cases the first letter of every chunk, which in a
phoneme model deletes or rewrites the first word. Swap back to the PyPI release
once the flag is upstream.
