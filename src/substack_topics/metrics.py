import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from itertools import combinations
from contextlib import contextmanager

import logging
log = logging.getLogger(__name__)

from .tokenise import simple_tokenise

# Computational efficiency metrics ---------------------------------------------------------------------------
@contextmanager
def track_runtime():
    """Context manager to measure runtime"""
    start = time.perf_counter()
    info = {'runtime_s': 0.0}
    yield info
    info['runtime_s'] = round((time.perf_counter() - start), 1)
    info['runtime_m'] = round(info['runtime_s'] / 60, 1)


# Helper functions ---------------------------------------------------------------------------
def _normalise(x) -> np.ndarray:
    """Normalizes arrays to unit length, safely handling both 1D and 2D shapes."""
    x = np.asarray(x, dtype=np.float32)
    
    # Dynamically determine the target axis and dimension keeping rule
    axis = 1 if x.ndim == 2 else 0
    keepdims = True if x.ndim == 2 else False
    
    # Calculate Euclidean length along the correct axis
    norm = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    
    # Return a new object: normalized input with a safety clip against division-by-zero
    return x / np.clip(norm, 1e-12, None)

 
# Structure-based metrics ---------------------------------------------------------------------------
def topic_diversity(topics: list[list[str]], topn: int = 10) -> float:
    """Share of unique words among the top-k words of all topics (1.0 = no overlap).

    Args:
        topics (list[list[str]]): List of topics, each represented as a list of words.
        topn (int, optional): Number of top words to consider from each topic. Defaults to 10.
    
    Returns:
        float: Topic diversity score (0.0 = complete overlap, 1.0 = no overlap).
    """
    words = [w for t in topics for w in t[:topn]]
    return len(set(words)) / len(words) if words else float("nan")


def outlier_rate(doc_topics: list[int], outlier: int = -1) -> float:
    """Proportion of documents that are outliers (assigned the outlier topic).

    Args:
        doc_topics (list[int]): List of topic assignments for each document.
        outlier (int, optional): The topic index considered as an outlier. Defaults to -1.
    
    Returns:
        float: Outlier rate (0.0 = no outliers, 1.0 = all outliers).
    """
    return float(np.mean(np.asarray(doc_topics) == outlier))

def get_topic_counts(doc_topics, total_topics: int) -> tuple[np.ndarray, int]:
    """Internal helper to calculate topic document counts and total assigned documents safely."""
    d = np.asarray(doc_topics)
    assigned_mask = d >= 0
    sizes = np.bincount(d[assigned_mask], minlength=total_topics) if assigned_mask.any() else np.zeros(total_topics, int)
    assigned = max(int(assigned_mask.sum()), 1)
    return sizes, assigned

def topic_size_stats(doc_topics, n_topics: int) -> dict:
    """Catches 'one giant topic' and 'dozens of tiny topics'. Outliers (-1) are excluded.
    
    Args:
        doc_topics (np.ndarray): Array of topic assignments for each document.
        n_topics (int): Total number of topics (excluding outliers).

    Returns:
        dict: Dictionary containing topic size statistics calculated relative 
              to the total number of assigned documents including 
              topic share, topic size and number of tiny topics (fewer than 0.1% of assigned documents, or fewer than 5 documents).
    """
    
    sizes, assigned = get_topic_counts(doc_topics, n_topics)
    total_docs = len(doc_topics)

    return {
        # --- Raw Sizes ---
        "largest_topic_size": int(sizes.max()),
        "median_topic_size": float(np.median(sizes)),
        "smallest_topic_size": int(sizes.min()),
        
        # --- Relative Shares (Normalized: Outliers Excluded) ---
        # Useful for checking how the model splits up the information it understood
        "largest_topic_relative_%": float(sizes.max() / assigned * 100),
        "median_topic_relative_%": float(np.median(sizes) / assigned * 100),
        "smallest_topic_relative_%": float(sizes.min() / assigned * 100),
        
        # --- Absolute Shares (Volume: Outliers Included) ---
        # Useful for understanding overall throughput and data-to-business impact
        "largest_topic_absolute_%": float(sizes.max() / total_docs * 100),
        "median_topic_absolute_%": float(np.median(sizes) / total_docs * 100),
        "smallest_topic_absolute_%": float(sizes.min() / total_docs * 100),
        
        # --- Structural Warning Check ---
        # Calculated against assigned documents
        "n_tiny_topics": int((sizes < max(0.001 * assigned, 5)).sum()),
    }

# Coherence-based metrics ---------------------------------------------------------------------------
def build_reference_tokens_and_embeddings(
    texts: list[str], 
    max_docs: int = 20_000, 
    seed: int = 0,
    reference_embeddings_model: str = "all-mpnet-base-v2",
) -> tuple[list[list[str]], np.ndarray]:
    """Subsamples and tokenises a dataset to use as a consistent coherence reference.
    
    Args:
        texts: A list of raw text strings.
        max_docs: The maximum number of documents to sample. Defaults to 20,000.
        seed: Random seed for reproducible sampling. Defaults to 0.
        reference_embeddings_model: The embedding model to use for reference documents. Defaults to "all-mpnet-base-v2".
        
    Returns:
        A tuple containing:
            - A list of tokenised reference documents (lists of words).
            - A Gensim dictionary built from the reference tokens.
            - An array of embeddings for the reference documents.
            - The indices of the sampled reference documents within the original texts.
    """
    from gensim.corpora import Dictionary

    rng = np.random.default_rng(seed)
    sample_size = min(max_docs, len(texts))
    
    # Randomly pick unique indices without replacement
    ref_idx = rng.choice(len(texts), size=sample_size, replace=False)
    
    # Tokenise only the selected documents
    ref_tokens = [simple_tokenise(texts[i]) for i in ref_idx]
    log.info(f"Extracted {len(ref_tokens)} reference tokenised documents for coherence evaluation.")

    ref_tokens_dictionary = Dictionary(ref_tokens)
    log.info(f"Built dictionary with {len(ref_tokens_dictionary)} unique tokens for coherence evaluation.")

    from .embeddings import get_embeddings
    ref_embeddings = get_embeddings([texts[i] for i in ref_idx], reference_embeddings_model)
    log.info(f"Computed embeddings for {len(ref_embeddings)} reference documents for coherence evaluation.")

    return ref_tokens, ref_tokens_dictionary, ref_embeddings, ref_idx


def coherence_scores(
    topics: list[list[str]], 
    tokenised_texts: list[list[str]],
    measures: tuple[str, ...] = ("c_v", "c_npmi"), 
    ref_tokens_dictionary = None, 
    window_size: int=100,
    processes: int = 1, 
    topn: int = 10,
) -> dict[str, float | None]:
    """Gensim coherence. Returns None values if gensim is not installed.

    Use the SAME tokenised reference texts for every model, otherwise scores are not comparable.
    Coherence on very short texts is noisy and does not always agree with human judgement.

    Args:
        topics (list[list[str]]): List of topics, each represented as a list of words.
        tokenised_texts (list[list[str]]): List of tokenised reference texts for coherence calculation.
        measures (tuple[str, ...], optional): Coherence measures to compute. Defaults to ("c_v", "c_npmi").
        ref_tokens_dictionary ("gensim.corpora.Dictionary" | None, optional): Gensim dictionary for the reference texts. If None, a new dictionary will be built from `tokenised_texts`. Defaults to None.
        window_size (int, optional): The window size for coherence calculation. Defaults to 100 as dataset consist of short texts.
        processes (int, optional): Number of processes to use for coherence calculation. Defaults to 4.
        topn (int, optional): Number of top words to consider from each topic. Defaults to 10.

    Returns:
        dict[str, float | None]: Dictionary mapping coherence measure names to their scores. None if gensim is not installed.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    # Build the dictionary if not provided
    if ref_tokens_dictionary is None:
        ref_tokens_dictionary = Dictionary(tokenised_texts)

    # Filter topics based on the topn words
    valid = set(ref_tokens_dictionary.token2id)

    # lowercase & dedupe, keeping order
    lowered = [list(dict.fromkeys(w.lower() for w in t)) for t in topics]   
    
    # Calculate expected slots vs found slots (Total slots = topics * topn)
    total_words_expected = sum(len(t[:topn]) for t in lowered)
    words_found = sum(w in valid for t in lowered for w in t[:topn])

    out = {
        "coherence_words_expected": total_words_expected if total_words_expected else None,
        "coherence_words_found": words_found if total_words_expected else None,
        "coherence_word_coverage": words_found / total_words_expected if total_words_expected else None,
    }

    # Keep only words that are in the reference dictionary and limit to topn
    tops = [[w for w in t if w in valid][:topn] for t in lowered]
    
    # Strict: only keep topics that have exactly topn valid words
    tops = [t for t in tops if len(t) == topn]
    # Only drop a topic if it has 0 valid words left
    # tops = [t for t in tops if len(t) > 0]

    out["coherence_n_topics_scored"] = len(tops)
    if not tops:
        return out | {f"coherence_{m}": None for m in measures}

    for m in measures:
        cm = CoherenceModel(
            topics=tops, 
            texts=tokenised_texts, 
            dictionary=ref_tokens_dictionary,
            coherence=m, 
            topn=topn, 
            window_size=window_size,  
            processes=processes       
        )
        out[f"coherence_{m}"] = float(cm.get_coherence())
    return out

def word_embedding_coherence(topics: list[list[str]], wv, topn: int = 10, lowercase: bool = True) -> dict:
    """
    Mean pairwise cosine similarity of a topic's top words, using pretrained word vectors. Independent of corpus co-occurrence.
    Like the gensim coherence, every topic is scored on exactly `topn` words: the first `topn` words found in
    the vectors (fit models with more words, e.g. 30). Topics that cannot reach `topn` are left out
    and counted. `wv` is a gensim KeyedVectors, e.g.
    
    Args:
        topics (list[list[str]]): List of topics, each a list of words.
        wv: Pretrained word vectors (gensim KeyedVectors).
        topn (int, optional): Number of top words to consider per topic. Defaults to 10.
        lowercase (bool, optional): Whether to lowercase words before checking in vectors. Defaults to True.

    Returns:
        dict: Dictionary containing word embedding coherence metrics.
    """
    per_topic: list[float] = []
    expected = found = 0
    for t in topics:
        words = list(dict.fromkeys(w.lower() if lowercase else w for w in t))
        first = words[:topn]
        expected += len(first)
        found += sum(w in wv.key_to_index for w in first)       
        usable = [w for w in words if w in wv.key_to_index][:topn]

        # Strict: only keep topics that have exactly topn valid words
        if len(usable) < topn:
            continue
        V = _normalise(np.asarray(wv[usable], dtype=np.float32))
        S = V @ V.T
        per_topic.append(float(S[np.triu_indices(topn, k=1)].mean()))
    return {
        "wec": float(np.mean(per_topic)) if per_topic else float("nan"),
        "wec_n_topics_scored": len(per_topic),
        "wec_word_coverage": found / expected if expected else float("nan"),
        "wec_per_topic": per_topic,
    }

# Document level Embedding-based metrics ---------------------------------------------------------------------------
def _centroid_scores(X: np.ndarray, labels: np.ndarray, outlier: int) -> tuple[float, list[float]]:
    """Compute centroid-based similarity scores for each topic.

    Args:
        X (np.ndarray): Normalized document embeddings.
        labels (np.ndarray): Topic labels for each document.
        outlier (int): Label indicating outlier documents to ignore.

    Returns:
        tuple[float, list[float]]: Overall mean similarity and per-topic mean similarities.
    """

    sims, per_topic = [], []
    unique_labels = np.unique(labels[labels != outlier])
    for t in unique_labels:
        Xt = X[labels == t]
        c = Xt.mean(axis=0)
        c = _normalise(c)
        s = Xt @ c                                  # cosine, since everything is unit length
        sims.append(s)
        per_topic.append(float(s.mean()))
    if not sims:
        return float("nan"), []
    return float(np.concatenate(sims).mean()), per_topic
 
 
def centroid_similarity(embeddings, doc_topics, outlier: int = -1, seed: int = 0) -> dict:
    """Mean cosine similarity between each document and the centroid of its topic.

    Args:
        embeddings (np.ndarray): Document embeddings.
        doc_topics (np.ndarray): Topic assignments for each document.
        outlier (int): Label indicating outlier documents to ignore.
        seed (int): Random seed for shuffling to compute chance similarity.
    
    Returns:
        dict: A dictionary containing centroid similarity metrics.
                - "centroid_sim": Overall average document-weighted cosine similarity, 
                - "centroid_sim_topic_mean": the unweighted mean over topics,
                - "centroid_sim_chance": Random benchmark obtained by shuffling topic labels among assigned documents.
                - "centroid_sim_diff": Main evaluation metric. The difference between the document-weighted mean and the chance score.
                - "centroid_sim_per_topic": List of per-topic centroid similarities, ordered by topic id. Ideal for spotting exactly which specific topic is leaking or drifting

    """
    X, d = _normalise(embeddings), np.asarray(doc_topics)
    mean_all, per_topic = _centroid_scores(X, d, outlier)
 
    shuffled = d.copy()
    assigned = d != outlier
    # Shuffle only the assigned documents while keeping cluster sizes exactly the same to compute chance similarity.
    shuffled[assigned] = np.random.default_rng(seed).permutation(d[assigned])
    chance, _ = _centroid_scores(X, shuffled, outlier)
    diff = mean_all - chance
 
    return {
        "centroid_sim": mean_all,
        "centroid_sim_topic_mean": float(np.mean(per_topic)) if per_topic else float("nan"),
        "centroid_sim_chance": chance,
        "centroid_sim_diff": diff,
        "centroid_sim_per_topic": per_topic,        # ordered by topic id
    } 

def silhouette(embeddings, doc_topics, outlier: int = -1, sample_size: int = 20_000, seed: int = 0) -> float:
    """
    Calculate the cosine silhouette coefficient for the assigned documents to be compared across models and not as an absolute measure.
    
    Args:
        embeddings (np.ndarray): Document embeddings.
        doc_topics (np.ndarray): Topic assignments for each document.
        outlier (int): Label indicating outlier documents to ignore.
        sample_size (int): Number of documents to sample for silhouette calculation.
        seed (int): Random seed for reproducibility.
    
    Returns:
        float: Cosine silhouette coefficient for the assigned documents. (-1 to 1; higher = tighter, better separated topics)
    """
    from sklearn.metrics import silhouette_score
 
    d = np.asarray(doc_topics)
    mask = d != outlier
    X, y = np.asarray(embeddings)[mask], d[mask]
    if len(np.unique(y)) < 2:
        return {"silhouette": float("nan")}
    return {
        "silhouette": float(silhouette_score(
                            X, y, metric="cosine",
                            sample_size=sample_size if len(y) > sample_size else None,
                            random_state=seed
                            ))
    }    


# Single-run evaluation metrics ---------------------------------------------------------------------------
def evaluate_topic_result(
        result, 
        texts: list[str], 
        max_docs: int = 20_000, 
        topn: int = 10,
        wv = None,  
        seed: int = 0,
        reference: tuple | None = None
) -> dict:
    """Evaluate a single TopicResult with various metrics.

    Args:
        result: The TopicResult object containing topics and document-topic assignments.
        texts: The list of tokenised documents.
        max_docs: Maximum number of documents to use for building reference tokens.
        topn: Number of top words per topic to consider.
        wv: Pretrained word vectors for word embedding coherence calculation.
        seed: Random seed for reproducibility.
        reference: A tuple containing reference tokens, dictionary, embeddings, and indices, if available.

    Returns:
        A dictionary of evaluation metrics.
    """

    rec: dict = {
        "n_topics": len(result.topics),
        "topic_diversity": topic_diversity(result.topics, topn),
        "outlier_rate": outlier_rate(result.doc_topics),
    }

    # Build reference tokens and dictionary for coherence calculation
    if reference is None:
        reference = build_reference_tokens_and_embeddings(texts, max_docs=max_docs, seed=seed)
    ref_tokens, ref_tokens_dictionary, ref_embeddings, ref_idx = reference

    rec.update(topic_size_stats(result.doc_topics, len(result.topics)))
    log.info("Calculating coherence scores...")
    rec.update(coherence_scores(result.topics, ref_tokens, ref_tokens_dictionary=ref_tokens_dictionary, topn=topn))
    log.info("Calculating word embedding coherence...")
    rec.update(word_embedding_coherence(result.topics, wv=wv, topn=topn))
    log.info("Calculating centroid similarity...")
    rec.update(centroid_similarity(ref_embeddings, result.doc_topics[ref_idx], outlier=-1, seed=seed))
    log.info("Calculating silhouette coefficient...")
    rec.update(silhouette(ref_embeddings, result.doc_topics[ref_idx], outlier=-1, seed=seed))
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()}

# Stability across random seeds ---------------------------------------------------------------------------
def mean_pairwise_ari(label_sets: list) -> float:
    """Mean adjusted Rand index between the document partitions of each pair of runs."""
    if len(label_sets) < 2:
        return float("nan")
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]))
 
 
def topic_overlap(topics_a: list[list[str]], topics_b: list[list[str]], topk: int = 10) -> float:
    """For each topic in A, the best Jaccard overlap of top words with any topic in B; averaged."""
    if not topics_a or not topics_b:
        return float("nan")
    sets_b = [set(t[:topk]) for t in topics_b]
    scores = []
    for t in topics_a:
        sa = set(t[:topk])
        scores.append(max(len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0 for sb in sets_b))
    return float(np.mean(scores))
 
 
def mean_topic_overlap(topic_sets: list[list[list[str]]], topk: int = 10) -> float:
    """Symmetric topic_overlap, averaged over every pair of runs."""
    if len(topic_sets) < 2:
        return float("nan")
    vals = [(topic_overlap(a, b, topk) + topic_overlap(b, a, topk)) / 2
            for a, b in combinations(topic_sets, 2)]
    return float(np.mean(vals))
 
 
def stability(results: list, topk: int = 10) -> dict:
    """Stability of one model across seeds. `results` = list of TopicResult, one per seed.
 
    Note: outliers (-1) count as one cluster in the ARI, so a model that dumps many documents
    into the outlier topic (BERTopic) can look more stable than it is. Read it with outlier_rate.
    """
    return {
        "stability_ari": mean_pairwise_ari([r.doc_topics for r in results]),
        "stability_topic_overlap": mean_topic_overlap([r.topics for r in results], topk),
    }

