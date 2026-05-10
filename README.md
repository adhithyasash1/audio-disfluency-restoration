# Hindi Disfluency Restoration

Restore filler words (`हम्म`, `हाँ`, `उम्म`, `तो`, …) to *clean* Hindi transcripts using a Hindi-tuned Whisper-Large-V3 ASR model, sequence alignment, a position prior, and a trigram language model.

This repository is the **source-only**, refactored version of the original Kaggle competition notebook. The pipeline is now a small Python package (`disfluency/`) with unit tests and a CLI; the original notebook is kept as the explanatory walkthrough.

---

## What it does

Given:
- a clean Hindi transcript with all filler words removed, and
- the original audio,

produce the transcript with the spoken fillers restored. Evaluated by Word Error Rate (WER).

## Workflow

```
clean text ──┐
              │     ┌──────────────────┐
   audio ───► │ ──► │ Whisper ASR      │ ──► tokens + per-word log-probs
              │     └──────────────────┘
              │              │
              ▼              ▼
       SequenceMatcher ── insert ops ──► filter (per-token threshold,
                                                position prior, trigram LM)
                                                       │
                                                       ▼
                                         apply right-to-left ──► restored text
```

| Module | Responsibility |
|--------|----------------|
| `disfluency.text` | Tokenization, normalization, Devanagari-safe filler stripping |
| `disfluency.thresholds` | Per-token Whisper log-prob thresholds |
| `disfluency.position` | Position prior (start-of-sentence bias) |
| `disfluency.ngram` | Trigram LM with add-α smoothing |
| `disfluency.align` | `SequenceMatcher` + decision logic |
| `disfluency.asr` | Whisper wrapper (uses `compute_transition_scores`) |
| `disfluency.cache` | Per-id JSON cache for ASR results |
| `disfluency.pipeline` | Orchestrator: `Restorer.restore()` |
| `disfluency.cli` | `disfluency-restore` entry point |

---

## Expected input format

The pipeline expects three artifacts that are **not committed to this repo** (see *Intentionally not included* below):

1. **`test.csv`** — header `id,transcript`. `transcript` is the clean Hindi text.
2. **`train.csv`** *(optional, recommended)* — header `id,transcript`. Used only to fit the trigram LM. Without it, the LM filter is disabled.
3. **`downloaded_audios/`** — one `{id}.wav` per row in `test.csv`, mono, any sample rate (resampled to 16 kHz internally).

Output is written to `outputs/submission.csv` (header `id,transcript`).

---

## Intentionally not included

To keep the repo lightweight, the following are **never committed** (see `.gitignore`):

- The Kaggle dataset (audio `.wav`, `train.csv`, `test.csv`, `unique_disfluencies.csv`).
- Whisper model weights (downloaded at runtime from Hugging Face).
- The ASR cache (`outputs/asr_cache/`) and any `submission.csv`.
- Virtualenvs, IDE metadata, `__pycache__`, build artifacts.

---

## Install

Requires Python ≥ 3.10. A GPU is strongly recommended for ASR; CPU works but is ~30× slower.

```bash
git clone https://github.com/adhithyasash1/audio-disfluency-restoration.git
cd audio-disfluency-restoration

python -m venv .venv && source .venv/bin/activate
pip install -e ".[asr,dev]"        # full pipeline + tests
# or, to skip the heavy ASR stack and only use the pure-Python helpers:
pip install -e .
```

---

## Run

### Inference (CLI)

```bash
disfluency-restore \
  --audio-dir /path/to/downloaded_audios \
  --test-csv  /path/to/test.csv \
  --train-csv /path/to/train.csv \
  --out       outputs/submission.csv \
  --cache-dir outputs/asr_cache
```

Useful flags: `--no-lm`, `--pos-exponent 1.5`, `--lm-threshold -2.0`, `--model <hf-id>`, `--seed 0`.

### Inference (Python)

```python
from disfluency.asr import WhisperASR
from disfluency.cache import AsrCache
from disfluency.ngram import NgramLM
from disfluency.pipeline import RestorationConfig, Restorer
import pandas as pd

train = pd.read_csv("train.csv")["transcript"].dropna().astype(str).tolist()
restorer = Restorer(
    asr=WhisperASR(),
    lm=NgramLM.from_transcripts(train),
    cache=AsrCache("outputs/asr_cache"),
    config=RestorationConfig(pos_exponent=1.5, use_lm=True, lm_threshold=-2.0),
)
print(restorer.restore("मैं ठीक हूं", audio_id="ex1", audio_path="ex1.wav"))
```

### Notebook walkthrough

[`submission.ipynb`](submission.ipynb) is the original Kaggle end-to-end notebook, kept as a self-contained explanation of the algorithm. The code in `disfluency/` is the production-quality re-implementation with bug fixes.

### Training

There is **no model training step** in this project — Whisper is used zero-shot via the public ARTPARK-IISc fine-tune; the only "training" is fitting the trigram LM from `train.csv`, which happens automatically when you pass `--train-csv`.

### Evaluate

If you have a held-out CSV with both clean and gold-restored columns:

```python
from jiwer import wer
print(wer(gold_list, predicted_list))
```

A held-out 90/10 split + ablation script is on the roadmap.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/submission.csv` | `id,transcript` with restored fillers |
| `outputs/asr_cache/{id}.json` | Cached ASR text + per-word log-probs |

Re-running with the cache populated skips Whisper entirely.

---

## Tests

```bash
pytest -q
```

Covers the pure-Python pieces (text/regex/alignment/position/n-gram). The ASR layer is not exercised in CI because it requires GPU + 3 GB of model weights.

---

## Troubleshooting

- **`No module named 'transformers'`** — install the ASR extra: `pip install -e ".[asr]"`.
- **Whisper download is slow** — `huggingface-cli download ARTPARK-IISc/whisper-large-v3-vaani-hindi` once, or pass `--model` to a locally-cached path.
- **CUDA OOM** — pass `--model openai/whisper-medium` or run on CPU; FP16 is auto-enabled on CUDA only.
- **Empty `submission.csv` rows** — audio paths don't resolve. Check that `{audio-dir}/{id}.wav` exists for every row in `test.csv`.
- **"forced_decoder_ids deprecated"** — upgrade transformers to ≥ 4.44.

---

## Roadmap

- [ ] Local 90/10 eval + ablation script
- [ ] Replace trigram LM with IndicBERTv2 masked-LM scoring
- [ ] HF chunked ASR pipeline with stride to fix boundary-word loss
- [ ] Learned classifier over `(logprob, position, mlm_score, neighbors)` instead of hand-tuned thresholds
- [ ] GitHub Actions: lint + tests on every push

---

## License

MIT.
