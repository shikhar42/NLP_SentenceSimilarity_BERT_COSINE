"""Tech-persona sentence similarity, rebuilt pipeline.

Replaces the original ``bert_serving`` (``BertClient``) implementation, which
embedded one word at a time in a loop of one ``encode`` call per
word-dictionary pair and scored each word by its *maximum* cosine against the
dictionary (so a message word identical to a dictionary entry scored 1.0).

This module instead:

* uses a pinned sentence-transformers checkpoint (see ``MODEL_NAME`` /
  ``MODEL_REVISION``) and embeds **whole messages**, so word context --
  including negation -- is kept;
* scores a message against a single dictionary *prototype* (the mean of the
  batched dictionary embeddings);
* keeps negations during preprocessing (NLTK's English stopword list
  contains "not" and "no", so the old code deleted them);
* computes leave-one-out **lift** by token *position*, so repeated words are
  handled individually instead of being collapsed by ``groupby('word')``;
* batches every embedding call (one ``encode`` per set of texts, never a
  per-pair loop).

Model: ``sentence-transformers/all-MiniLM-L6-v2`` -- a MiniLM-L6 encoder
fine-tuned for sentence similarity. Pooling is the checkpoint's configured
**mean pooling** over token embeddings (sentence-transformers default for
this checkpoint), and embeddings are L2-normalised, so cosine similarity is
a dot product. No TensorFlow / bert-serving server is needed any more; the
stack is PyTorch via sentence-transformers (see requirements.txt).

Functions
---------
preprocess(message) -> list[str]
    Lowercase, tokenise, drop punctuation and stopwords *except* negations.
persona(message, encoder, dictionary_embeddings) -> float
    Whole-message interest score.
lift_by_position(message, encoder, dictionary_embeddings) -> pandas.DataFrame
    Per-position leave-one-out lift.
tfidf_scores(messages, dictionary) -> numpy.ndarray
    TF-IDF cosine baseline.
averaged_word_vector_scores(messages, encoder, dictionary_embeddings)
    Averaged-word-vector baseline (words embedded in one batched call).
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Model (pinned)
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned revision (commit) of the checkpoint on the Hugging Face Hub, so runs
# are reproducible even if the model's default branch changes.
MODEL_REVISION = "1110b5dce11483d046163640f88898a541d1a48a"
POOLING = "mean pooling over token embeddings (checkpoint default), L2-normalised"

# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------
# The original list had 59 entries. Fixes applied here:
#   * duplicate "electronics" removed               -> 58 unique entries
#   * "headhopnes" corrected to "headphones"
#   * case normalised ("TV" -> "tv", "Camera" -> "camera", "IOT" -> "iot",
#     "Home Audio" -> "home audio", "Computer Parts" -> "computer parts")
# The notebook text previously said "approximately 50" words; the correct
# count is 59 entries / 58 unique, and the README/notebook now say so.
DICTIONARY = [
    "tech", "gta 4", "gta", "gaming keyboard", "pubg", "fifa", "technology",
    "laptop", "mouse", "electronics", "desktop", "tv", "television",
    "home audio", "apps", "cloud", "web", "interface", "innovation",
    "machine learning", "artificial intelligence", "sensor", "data science",
    "data", "charger", "git", "version", "update", "phones", "home theater",
    "gadgets", "iot", "computer science", "camera", "photo", "video",
    "cell phone", "mobile", "headphones", "games", "video games", "gaming",
    "bluetooth", "wireless", "speakers", "amazon", "tablets", "software",
    "printers", "drives", "floppy", "keyboard", "apple", "iphone", "android",
    "samsung", "computer parts", "age of empires",
]
assert len(DICTIONARY) == 58 and len(set(DICTIONARY)) == 58

# ---------------------------------------------------------------------------
# Preprocessing -- negations are KEPT
# ---------------------------------------------------------------------------
# NLTK's English stopwords include "not", "no" and "nor". Removing them (as
# the original code did) flips meaning: "I do not like gaming" would score
# like "I like gaming". Negations are therefore excluded from the stopword
# set used here.
NEGATIONS = {"not", "no", "nor", "never", "n't", "cannot"}
STOPWORDS = set(stopwords.words("english")) - NEGATIONS


def preprocess(message: str) -> list[str]:
    """Tokenise ``message`` for the word-level baselines / lift analysis.

    :param message: input message text
    :returns: filtered tokens (lowercased, punctuation and stopwords
              removed, negations kept)

    Note: the transformer score itself is computed from the *raw* message
    (see :func:`persona`), because the encoder uses stopwords and word order
    for context. This tokenisation is used for the TF-IDF / averaged-vector
    baselines and for leave-one-out lift.
    """
    tokens = word_tokenize(message.lower())
    return [t for t in tokens if t not in string.punctuation and t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Embedding helpers -- batched
# ---------------------------------------------------------------------------

def load_encoder():
    """Load the pinned sentence-transformers model (downloads once, cached)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)


def embed_texts(encoder, texts: list[str]) -> np.ndarray:
    """Embed a list of texts in ONE batched ``encode`` call."""
    return np.asarray(
        encoder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )


def dictionary_prototype(encoder, dictionary: list[str] | None = None) -> np.ndarray:
    """Mean embedding of the dictionary entries (one batched encode call)."""
    dictionary = DICTIONARY if dictionary is None else dictionary
    emb = embed_texts(encoder, list(dictionary))
    prototype = emb.mean(axis=0)
    norm = np.linalg.norm(prototype)
    return prototype / norm if norm else prototype


# ---------------------------------------------------------------------------
# Interest score -- whole message, not single words
# ---------------------------------------------------------------------------

def persona(message: str, encoder, prototype: np.ndarray) -> float:
    """Interest score of a whole message against the dictionary prototype.

    :param message: input message text
    :param encoder: a sentence-transformers model (see :func:`load_encoder`)
    :param prototype: dictionary prototype from :func:`dictionary_prototype`
    :returns: cosine similarity in [-1, 1] between the embedding of the
              whole message (context and negations intact) and the
              dictionary prototype
    """
    emb = embed_texts(encoder, [message])[0]
    return float(cosine_similarity(emb.reshape(1, -1), prototype.reshape(1, -1))[0, 0])


def interest_scores(messages: list[str], encoder, prototype: np.ndarray) -> np.ndarray:
    """Batched interest scores for a list of messages (one encode call)."""
    emb = embed_texts(encoder, list(messages))
    return cosine_similarity(emb, prototype.reshape(1, -1)).ravel()


# ---------------------------------------------------------------------------
# Lift -- by token position, so repeated words are handled individually
# ---------------------------------------------------------------------------

def lift_by_position(message: str, encoder, prototype: np.ndarray) -> pd.DataFrame:
    """Leave-one-out lift for each token *position* in ``message``.

    The original code grouped by the word string (``groupby('word')``) and
    removed words by value (``list.remove``), so a repeated word was scored
    once and removing it removed only / attributed the wrong occurrence.
    Here every position is removed by index and reported separately.

    :param message: input message text
    :param encoder: a sentence-transformers model
    :param prototype: dictionary prototype from :func:`dictionary_prototype`
    :returns: DataFrame with columns ``position``, ``word`` and ``lift``
              (baseline interest score minus the score without that token;
              positive lift = the token raised the interest score)
    """
    tokens = preprocess(message)
    if not tokens:
        return pd.DataFrame(columns=["position", "word", "lift"])
    baseline = persona(message, encoder, prototype)
    # Build every leave-one-out variant and embed them in ONE batched call.
    variants = [
        " ".join(tok for j, tok in enumerate(tokens) if j != i)
        for i in range(len(tokens))
    ]
    variant_scores = interest_scores(variants, encoder, prototype)
    return pd.DataFrame(
        {
            "position": list(range(len(tokens))),
            "word": tokens,
            "lift": baseline - variant_scores,
        }
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def tfidf_scores(messages: list[str], dictionary: list[str] | None = None) -> np.ndarray:
    """TF-IDF cosine baseline.

    The vectoriser is fitted on the dictionary entries; each message is
    scored by its maximum cosine against a dictionary entry's TF-IDF vector.
    """
    dictionary = DICTIONARY if dictionary is None else dictionary
    vectorizer = TfidfVectorizer()
    dict_vecs = vectorizer.fit_transform(list(dictionary))
    msg_vecs = vectorizer.transform(list(messages))
    return cosine_similarity(msg_vecs, dict_vecs).max(axis=1)


def averaged_word_vector_scores(
    messages: list[str], encoder, prototype: np.ndarray
) -> np.ndarray:
    """Averaged-word-vector baseline.

    Each message's preprocessed words are embedded in ONE batched call,
    averaged, and compared with the same dictionary prototype. Unlike the
    main score this discards word order and context -- that contrast is the
    point of the baseline.
    """
    token_lists = [preprocess(m) for m in messages]
    flat = [tok for toks in token_lists for tok in toks]
    if not flat:
        return np.zeros(len(messages))
    flat_emb = embed_texts(encoder, flat)  # single batched call for all words
    scores, start = [], 0
    for toks in token_lists:
        if not toks:
            scores.append(0.0)
            continue
        vec = flat_emb[start : start + len(toks)].mean(axis=0)
        start += len(toks)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm else vec
        scores.append(float(cosine_similarity(vec.reshape(1, -1), prototype.reshape(1, -1))[0, 0]))
    return np.asarray(scores)
