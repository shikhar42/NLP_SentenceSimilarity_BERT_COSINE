# NLP Sentence Similarity — Tech Persona (BERT / sentence-transformers)

> Coursework project (individual coding challenge). Rebuilt to fix the
> methodological problems in the original submission; the original examples
> and their scores are discussed in the notebook's corrected conclusion.

## Problem

Score how "technical" a message is for a tech-persona: given a short message,
produce an interest score by comparing it with a dictionary of tech terms,
and explain the score with per-word **lift** (how much the score drops when a
word is removed).

## Data

- **Dictionary:** 59 entries originally, **58 unique** after removing the
  duplicate "electronics"; "headhopnes" corrected to "headphones" and case
  normalised ("TV" → "tv", "Camera" → "camera"). Defined in
  `sentence_similarity.py`.
- **Evaluation dataset:** 128 hand-labelled messages (64 tech / 64
  non-tech), defined in the notebook. It was expanded from 48 in round 2:
  with 48 messages the test split held only 10, so accuracy moved in steps
  of 0.1 and the validation-chosen threshold was unstable. No message is a
  dictionary entry, and the dictionary is never part of the evaluation
  data — it is only used to build the prototype vector (an assertion in
  the notebook enforces the separation, and message uniqueness). Split
  (stratified, `random_state=42`): 76 train / 26 validation / 26 test. The
  decision threshold is chosen on validation only; the test split is
  scored once. 128 hand-written messages is still a small sample — the
  metrics below should be read as indicative, with wide uncertainty.

## Methods

- **Main score:** the *whole message* is embedded with a pinned
  sentence-transformers checkpoint and compared (cosine) with the dictionary
  prototype (mean of the dictionary embeddings).
  - Checkpoint: `sentence-transformers/all-MiniLM-L6-v2`,
    revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (pinned in code;
    round 2 corrected an earlier pin that was not a valid revision of the
    model repo and prevented any download).
  - Pooling: the checkpoint's configured **mean pooling** over token
    embeddings; embeddings are L2-normalised.
  - All embedding is **batched** (one `encode` call per set of texts). The
    original code made one encode call per word–dictionary pair via a
    `bert_serving` server; that dependency is removed entirely.
- **Baselines:** (1) TF-IDF cosine against a dictionary **prototype** (the
  centroid of the dictionary entries' TF-IDF vectors — round 2 replaced a
  maximum-over-entries score, under which a message containing one exact
  dictionary word scored 1.0, the same artefact round 1 removed from the
  transformer side); (2) averaged word vectors (same encoder, words
  averaged, scored against the same prototype — discards word
  order/context).
- **Preprocessing:** lowercasing, tokenisation, punctuation/stopword removal
  for the baselines and lift — with **negations kept** ("not"/"no" are in
  NLTK's stopword list and the original code deleted them, flipping meaning).
  The transformer itself reads the raw message, so context is preserved.
- **Lift:** leave-one-out by token **position**, so repeated words get
  separate lift values (the original grouped by the word string). The
  baseline and every leave-one-out variant are scored on the **same
  preprocessed token list** (round-2 fix): an earlier version scored the
  baseline on the raw message while the variants used preprocessed tokens,
  so every lift carried a constant preprocessing offset and the sign of a
  single word's lift was unreliable.
- **Metrics:** accuracy, ROC AUC and F1 on the held-out test set, plus the
  majority-class accuracy as a reference. All numbers are computed by the
  notebook when it runs — none are hard-coded; the Results table below is
  copied from a verified end-to-end run (the notebook's saved outputs show
  the same figures).

## Results

Measured by actually running the notebook end-to-end (round 2) with the
pinned stack — encoder: `sentence-transformers/all-MiniLM-L6-v2` at
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, sentence-transformers
3.0.1 / torch 2.3.1 (CPU). Dataset: the 128-message set above; thresholds
chosen on the 26-message validation split (best F1); test split (26
messages) scored once. The notebook's saved outputs show the same numbers.

| Method | Val. threshold | Test accuracy | Test AUC | Test F1 |
|---|---|---|---|---|
| Averaged word vectors (same encoder) | 0.658 | **0.962** | **0.994** | **0.963** |
| Transformer (whole message) | 0.136 | 0.808 | 0.911 | 0.783 |
| TF-IDF cosine (prototype) | 0.000 | 0.500 | 0.837 | 0.667 |
| Majority class (reference) | — | 0.500 | — | — |

Read honestly: on this small hand-written dataset the simple averaged
word-vector baseline *beats* the whole-message transformer score, and the
TF-IDF prototype scores rank the classes reasonably (AUC 0.837) but its
validation-chosen threshold (0.000) classifies everything as tech, hence
accuracy/F1 no better than the majority class. With 26 test messages each
message is worth ~3.8 accuracy points, so treat the gaps as indicative —
the notebook's conclusion discusses this. The original submission's
claimed ranges ("85%+ for tech, 75–82% for non-tech") did **not** hold on
its own examples — a non-tech message scored 82.35% — because the original
per-word maximum-cosine design inflated every score; the notebook's
conclusion explains this in full.

## How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
jupyter notebook codingchallenge_capstone_shikharchhabra.ipynb
```

The first run downloads the model checkpoint (~90 MB) from Hugging Face and
caches it. No TensorFlow or bert-serving server is required any more — the
original stack (Python 3.7 + TensorFlow + a separately running
`bert-serving` server) has been replaced by PyTorch via sentence-transformers.

## Versions

- Python **3.11** (developed/validated for syntax on 3.11–3.12).
- See `requirements.txt` for pinned package versions
  (sentence-transformers, torch, scikit-learn, nltk, pandas, matplotlib).
- TensorFlow: **not used** in the rebuilt version.

## Files

- `codingchallenge_capstone_shikharchhabra.ipynb` — main notebook.
- `sentence_similarity.py` — dictionary, preprocessing, scoring, lift,
  baselines.
- `requirements.txt`, `README.md`.
