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
- **Evaluation dataset:** 48 hand-labelled messages (24 tech / 24 non-tech),
  defined in the notebook. No message is a dictionary entry, and the
  dictionary is never part of the evaluation data — it is only used to build
  the prototype vector. Split (stratified, `random_state=42`):
  60% train / 20% validation / 20% test. The decision threshold is chosen on
  validation only; the test split is scored once.

## Methods

- **Main score:** the *whole message* is embedded with a pinned
  sentence-transformers checkpoint and compared (cosine) with the dictionary
  prototype (mean of the dictionary embeddings).
  - Checkpoint: `sentence-transformers/all-MiniLM-L6-v2`,
    revision `1110b5dce11483d046163640f88898a541d1a48a` (pinned in code).
  - Pooling: the checkpoint's configured **mean pooling** over token
    embeddings; embeddings are L2-normalised.
  - All embedding is **batched** (one `encode` call per set of texts). The
    original code made one encode call per word–dictionary pair via a
    `bert_serving` server; that dependency is removed entirely.
- **Baselines:** (1) TF-IDF cosine against the dictionary; (2) averaged word
  vectors (same encoder, words averaged — discards word order/context).
- **Preprocessing:** lowercasing, tokenisation, punctuation/stopword removal
  for the baselines and lift — with **negations kept** ("not"/"no" are in
  NLTK's stopword list and the original code deleted them, flipping meaning).
  The transformer itself reads the raw message, so context is preserved.
- **Lift:** leave-one-out by token **position**, so repeated words get
  separate lift values (the original grouped by the word string).
- **Metrics:** accuracy, ROC AUC and F1 on the held-out test set, plus the
  majority-class accuracy as a reference. All numbers are computed by the
  notebook when it runs — none are hard-coded.

## Results

Run the notebook: it prints the validation-chosen threshold per method and a
test-set table (accuracy / AUC / F1) for the transformer score and both
baselines. The original submission's claimed ranges ("85%+ for tech,
75–82% for non-tech") did **not** hold on its own examples — a non-tech
message scored 82.35% — because the original per-word maximum-cosine design
inflated every score; the notebook's conclusion explains this in full.

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
