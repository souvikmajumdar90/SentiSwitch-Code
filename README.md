# SentiSwitch

A sentiment-controlled code-mixed (Hinglish) natural language generation
model: **Algorithm 2** (Train Unified Switching-Aware SC-NLG Model) trained
and packaged for standalone inference and evaluation, using **Algorithm 3**
(Switching-Aware Sentiment-Controlled Beam Search) at generation time.

## ⚠️ Important: read this before using the checkpoint

The requested repo layout named the checkpoint `mt5_sentiswitch.pt`, implying
an mT5-based model. **The actual trained model is a lightweight custom GRU
encoder-decoder with attention** (not mT5, not any pretrained transformer
backbone) — see `checkpoints/model_config.json` for the exact architecture.
The checkpoint file here is named `sentiswitch_gru.pt` instead, so the
filename doesn't misrepresent what's inside it. If you specifically need an
mT5-based SentiSwitch, that would require retraining from an actual
pretrained mT5 checkpoint (e.g. via HuggingFace `transformers`), which is a
substantially larger undertaking than what's implemented here.

## What's real vs. adapted in the training data

This model was trained on **L3Cube-HingLID**
([l3cube-pune/code-mixed-nlp](https://github.com/l3cube-pune/code-mixed-nlp)):

| | Status |
|---|---|
| Sentences and per-token Hindi/English language-ID tags | **Real** — human-annotated |
| Sentiment labels (`pos`/`neg`/`neu`) | **Heuristic** — keyword-lexicon assigned, not verified gold labels |
| Context/response pairing | **Synthetic** — consecutive corpus sentences, not real dialogue turns |
| Train/test split | 80/20, held-out, no leakage (144 train / 36 test) |

Only 144 training examples were used — expect the model to generalize
poorly. This repo demonstrates a **working pipeline**, not a benchmark-grade
model. See `checkpoints/model_config.json` → `n_train_examples`.

## Repo structure

```
SentiSwitch/
├── checkpoints/
│   ├── sentiswitch_gru.pt      # trained model weights (renamed from
│   │                            # requested mt5_sentiswitch.pt -- see above)
│   ├── vocab.json               # context/response/lang/sentiment vocabularies
│   └── model_config.json        # architecture hyperparameters + provenance
├── data/
│   ├── hinglid_D_raw.json       # source (context, response, sentiment) data
│   └── test_set.json            # held-out test split used by evaluate.py
├── train_and_export.py          # regenerates the checkpoint from scratch
├── inference.py                 # Algorithm 3: beam search generation
├── evaluate.py                  # MIPE / BLEU / ROUGE-1/2/L on held-out data
├── requirements.txt
├── sample_test_inputs.json
├── sample_outputs.json          # REAL outputs from this checkpoint (not fabricated)
├── README.md
└── LICENSE
```

`data/` and `train_and_export.py` aren't in the originally requested tree
but are included because `inference.py`/`evaluate.py` can't function without
the vocab/config artifacts, and the checkpoint needs to be reproducible from
source rather than appearing from nowhere.

## Quick start

```bash
pip install -r requirements.txt
```

### Run inference on your own inputs

```bash
python3 inference.py --input sample_test_inputs.json --output my_outputs.json
```

Input format:
```json
[{"context": "kaisa laga movie dekh kar", "sentiment": "pos"}]
```
`sentiment` must be `pos`, `neg`, or `neu`.

Useful flags: `--beam_size`, `--max_len`, `--alpha` (sentiment weight),
`--beta` (CMI weight), `--gamma` (switch-consistency weight), `--cmi_target`.

### Evaluate on the held-out test set

```bash
python3 evaluate.py --test_file data/test_set.json
```

Prints MIPE / BLEU / ROUGE-1/2/L and writes per-example detail to
`evaluation_results.json`.

**Actual numbers from this checkpoint** (36 held-out examples):

| Metric | Value |
|---|---|
| MIPE | 0.2182 |
| BLEU | 0.0180 |
| ROUGE-1 | 0.1291 |
| ROUGE-2 | 0.0000 |
| ROUGE-L | 0.1177 |

These are low in absolute terms — expected given the 144-example training
set. MIPE dropping from ~0.99 on training data to 0.22 on held-out data (in
earlier experiments during development) indicates the sentiment classifier
head overfits badly at this data scale.

### Retrain from scratch

```bash
python3 train_and_export.py
```

Rebuilds the dataset split, retrains for 40 epochs, and re-exports
`checkpoints/` and `data/test_set.json`. Requires `data/hinglid_D_raw.json`
(included) or regenerate it from the raw L3Cube-HingLID `train.txt` — see
the wider pipeline notebook for that dataset-construction step.

## MIPE metric definition

MIPE is **not** a standardized published metric. It's defined here as:

```
MIPE = 1 - |1 - p(target_sentiment)|
```

using the model's own `sent_classifier` head probability for the requested
sentiment class, averaged over the evaluation set. Higher is better
(1.0 = model always confidently assigns full probability to the correct
class). If you have access to the original paper/formula this repo layout
implies, substitute that definition into `evaluate.py`'s `mipe_score()`
function.

## License

Code: MIT (see `LICENSE`). This does **not** cover the underlying datasets
(L3Cube-HingLID, and any data used to extend this pipeline) — check their
original sources for redistribution terms.
