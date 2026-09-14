# Listening test

Persian does not write the ezafe — a short linking vowel that carries grammar.
So a transcription of correct Persian and a transcription of the same speech
with every ezafe removed are the *same characters*, and word error rate cannot
represent the difference, let alone score it. The same holds for anything else
your orthography omits: Arabic and Hebrew short vowels, Japanese pitch accent,
tone in unmarked scripts.

This is the instrument that replaces the metric there. A native ear, on
sentences chosen to provoke the failures, scored blind.

## Use

```bash
python build_listening_test.py build \
    --model v1=hf://mehdi-hf/pocket-tts-farsi/farsi.yaml \
    --model v2=hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml \
    --voice prompt.wav --out listening

open listening/index.html          # score, then Save scores
python build_listening_test.py tally --out listening
```

One model is scored on its own terms; two are scored blind against each other.
It detects whether a model reads script or phonemes from its tokenizer and feeds
it accordingly, so mixed comparisons work. Keep the voice prompt at or under
five seconds — longer is outside the training window and the model tends to
continue the prompt instead of speaking the text.

## The set

34 sentences in [`listening_set.jsonl`](listening_set.jsonl), each tagged with
what to listen for.

| category | n | what it provokes |
|---|---|---|
| stop-initial | 8 | the first-word defect, one per stop or nasal onset |
| ezafe | 6 | chains up to four links deep |
| pause | 4 | commas, which G2P discards |
| number | 4 | digits against spelled-out forms |
| foreign | 3 | proper nouns that fragment into subword pieces |
| long | 3 | chunk seams and second-sentence openings |
| ezafe-absent | 2 | control: a *spurious* ezafe is the failure |
| glottal-control | 2 | control: should never fail |
| question | 2 | final intonation |

The two control categories carry more weight than their count suggests. Without
`ezafe-absent`, a model that sprays ezafe everywhere scores as well as one that
places it correctly. Without `glottal-control`, you cannot tell a bad session
from a bad model.

## Scoring

Four axes, scored separately rather than averaged, because these failures have
different causes and a single mean-opinion score hides which one moved:

- **ezafe** — every link present, none invented
- **first word** — the opening word is fully there
- **phrasing** — pauses fall where the punctuation is
- **naturalness** — would pass as a person reading

Leave an axis blank when the sentence does not test it. Most sentences probe
one or two.

## Result, v1 against v2

136 judgements, one native listener, models hidden and reshuffled per item:

| | v2 wins | tie | v1 wins |
|---|---|---|---|
| ezafe | 21 | 12 | 1 |
| first word | 22 | 12 | 0 |
| phrasing | 24 | 9 | 1 |
| naturalness | 26 | 5 | 3 |

Blinding was not decoration. The listener expected v1 to win, and it lost 5 of
136. An unblinded comparison had previously produced the opposite conclusion,
and that conclusion reached both model cards before this test corrected it.
