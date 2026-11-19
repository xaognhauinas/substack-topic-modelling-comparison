"""Four topic models behind one interface.

Every fit_* function takes (texts, params, seed) and returns a TopicResult, so the
evaluation code never needs to know which model produced it.

Heavy libraries (gensim, bertopic, turftopic) are imported inside each function:
`import models` stays fast, and NMF works even if the others are not installed.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable
 
import numpy as np
 
from substack_topics.metrics import get_topic_counts
 
log = logging.getLogger(__name__)

@dataclass
class TopicResult:
    doc_topics: np.ndarray        # dominant topic id per document; -1 = outlier / no vocabulary
    topics: list[list[str]]       # top words per topic; list index == topic id; outlier topic excluded
    extra: dict = field(default_factory=dict)   # the fitted model object, anything else worth keeping


def sort_topic_result(doc_topics, topics, outlier: int = -1) -> tuple[np.ndarray, list[list[str]]]:
    """Sorts topics and document assignments by size from largest to smallest.

    Maintains negative outlier indices exactly where they are. Uses get_topic_counts
    for internal size calculations.
    """
    d = np.asarray(doc_topics, dtype=int)
    n_topics = len(topics)
    
    # 1. Reuse your existing function to get counts
    sizes, _ = get_topic_counts(d, total_topics=n_topics)
    
    # 2. Determine sorting order: largest topic indices first
    sort_order = np.argsort(sizes)[::-1]
    
    # 3. Create a clean mapping array to translate old IDs to sorted IDs
    mapper = np.empty(n_topics, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        mapper[old_idx] = new_idx
        
    # 4. Safely map assigned rows and re-apply outlier definitions
    assigned_mask = d >= 0
    sorted_doc_topics = np.where(assigned_mask, mapper[np.clip(d, 0, None)], outlier)
    
    # 5. Rearrange the top keyword lists to match the new ID layout
    sorted_topics = [topics[old_idx] for old_idx in sort_order]
    
    return sorted_doc_topics, sorted_topics


# Bag-of-words models ---------------------------------------------------------------------------
def fit_nmf(texts: list[str], params: dict, seed: int) -> TopicResult:
    """TF-IDF + NMF. Fast baseline: every other model should beat it."""
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        stop_words="english",
        min_df=params.get("min_df", 5),
        max_df=params.get("max_df", 0.5),
        max_features=params.get("max_features", 50_000),
        ngram_range=tuple(params.get("ngram_range", (1, 2))),
    )
    X = vec.fit_transform(texts)
    log.info("Document-Term (TF-IDF) matrix shape: %s", X.shape)

    nmf = NMF(
        n_components=params.get("n_topics", 20),
        init=params.get("init", "nndsvda"),
        max_iter=params.get("max_iter", 300),
        random_state=seed,
    )
    W = nmf.fit_transform(X)
    log.info("NMF Document-Topic (W) matrix shape: %s", W.shape)

    # Post-process: Extract key words for each topic
    vocab = vec.get_feature_names_out()
    topn = params.get("top_n_words", 10)
    raw_topics = [[str(w) for w in vocab[np.argsort(comp)[::-1][:topn]]] for comp in nmf.components_]
    log.info("Extracted %d topics with top %d words each", len(raw_topics), topn)

    # Post-process: Assign each document to the topic with the highest weight
    doc_topics = W.argmax(axis=1)
    doc_topics[W.sum(axis=1) == 0] = -1     # no known vocabulary

    # Sort topics and reassign document-topic labels from biggest to smallest topic by document count
    doc_topics, topics = sort_topic_result(doc_topics, raw_topics, outlier=-1)
    log.info("Max topic probability for first 5 documents: %s", np.max(W, axis=1)[:5])
    log.info("Document-topic assignments shape: %s", doc_topics.shape)
    return TopicResult(doc_topics, topics, {"model": nmf, "vectorizer": vec})


def fit_lda(texts: list[str], params: dict, seed: int) -> TopicResult:
    """Gensim LDA on spaCy lemmas (nouns, proper nouns, adjectives)."""
    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore
    from gensim.matutils import corpus2csc

    from .tokenise import tokenise_for_lda

    tokens = tokenise_for_lda(texts, add_bigrams=params.get("add_bigrams", False))

    # Create dictionary and bag-of-words representation for LDA
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=params.get("no_below", 5), no_above=params.get("no_above", 0.5))
    bow = [dictionary.doc2bow(t) for t in tokens]
    log.info("LDA dictionary size: %d", len(dictionary))

    k = params.get("n_topics", 20)
    model = LdaMulticore(
        corpus=bow,
        id2word=dictionary,
        num_topics=k,
        passes=params.get("passes", 5),
        alpha=params.get("alpha", "symmetric"),
        eta=params.get("eta", None),                       
        workers=params.get("workers", 2),
        random_state=seed,
    )

    # Post-process: Extract key words for each topic
    topn = params.get("top_n_words", 10)
    raw_topics = [[w for w, _ in model.show_topic(i, topn=topn)] for i in range(k)]
    log.info("Extracted %d topics with top %d words each", len(raw_topics), topn)

    # Post-process: Assign each document to the topic with the highest probability
    topic_matrix = corpus2csc(model.get_document_topics(bow, minimum_probability=0.0)).T.toarray()
    doc_topics = np.argmax(topic_matrix, axis=1)
    is_empty_doc = np.array([len(b) == 0 for b in bow]) # mark empty documents to assign -1 topic
    doc_topics[is_empty_doc] = -1

    # Sort topics and reassign document-topic labels from biggest to smallest topic by document count
    doc_topics, topics = sort_topic_result(doc_topics, raw_topics, outlier=-1)
    log.info("Max topic probability for first 5 documents: %s", np.max(topic_matrix, axis=1)[:5])
    log.info("Document-topic assignments shape: %s", doc_topics.shape)
    return TopicResult(doc_topics, topics, {"model": model, "dictionary": dictionary})


# Embedding-based models ---------------------------------------------------------------------------
def fit_bertopic(texts: list[str], params: dict, seed: int) -> TopicResult:
    """BERTopic on cached sentence embeddings. Gets near-raw text: the encoder uses context."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    from .embeddings import get_embeddings

    emb_name = params.get("embedding_model", "all-MiniLM-L6-v2")
    emb = get_embeddings(texts, emb_name)
    log.info("Computed embeddings for %d documents", len(emb))
    log.info("Embedding dimension: %d", emb.shape[1])

    umap_model = UMAP(
        n_neighbors=params.get("umap_n_neighbors", 15),
        n_components=params.get("umap_n_components", 5),
        min_dist=0.0, 
        metric="cosine", 
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.get("min_cluster_size", 50),
        min_samples=params.get("min_samples", None),
        metric="euclidean", 
        cluster_selection_method="eom", 
        prediction_data=True,
    )
    # stop words are removed only here, in the vectoriser that labels the topics
    vectorizer = CountVectorizer(
        stop_words="english", 
        ngram_range=tuple(params.get("ngram_range", (1, 2))), 
        min_df=params.get("min_df", 5))

    topn = params.get("top_n_words", 10)
    model = BERTopic(
        embedding_model=emb_name,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=params.get("nr_topics", 20),          
        top_n_words=topn,
        calculate_probabilities=False,
        verbose=True,
    )
    raw, _ = model.fit_transform(texts, emb)

    # Post-process: Re-indexing topic IDs to be contiguous 0..K-1
    ids = sorted(t for t in model.get_topics() if t != -1)
    remap = {old: new for new, old in enumerate(ids)}

    # Post-process: Extract key words for each topic
    topics = [[w for w, _ in model.get_topic(t)][:topn] for t in ids]
    log.info("Extracted %d topics with top %d words each", len(topics), topn)

    # Post-process: Map raw topic assignments to contiguous topic IDs
    doc_topics = np.array([remap.get(t, -1) for t in raw], dtype=int)
    log.info("Document-topic assignments shape: %s", doc_topics.shape)
    return TopicResult(doc_topics, topics, {"model": model, "embeddings": emb})

# Factorization models ---------------------------------------------------------------------------
def fit_keynmf(texts: list[str], params: dict, seed: int) -> TopicResult:
    """Turftopic's KeyNMF on the same cached embeddings BERTopic uses.
 
    Per the Turftopic docs: the encoder is passed as a SentenceTransformer object, the
    precomputed embeddings must come from that same model, and fit_transform returns a
    (documents x topics) matrix. Still unverified: the shape returned by get_topics().
    """
    from sentence_transformers import SentenceTransformer
    from turftopic import KeyNMF
 
    from .embeddings import get_embeddings
 
    enc_name = params.get("embedding_model", "all-MiniLM-L6-v2")
    emb = get_embeddings(texts, enc_name)
    log.info("Computed embeddings for %d documents", len(emb))
    log.info("Embedding dimension: %d", emb.shape[1])
 
    model = KeyNMF(
        params.get("n_components", 20),
        encoder=SentenceTransformer(enc_name),
        random_state=seed,
        
    )
    doc_topic_matrix = model.fit_transform(texts, embeddings=emb)
    log.info("Document-topic matrix shape: %s", doc_topic_matrix.shape)
    log.info("Max topic probability for first 5 documents: %s", np.max(doc_topic_matrix, axis=1)[:5])

    topn = params.get("top_n_words", 10)
    # Post-process: Extract key words for each topic
    raw_topics = [[w for w, _ in words] for _, words in model.get_topics(top_k=topn)]
    log.info("Extracted %d topics with top %d words each", len(raw_topics), topn)

    # Post-process: Assign each document to the topic with the highest probability
    doc_topic_matrix = np.asarray(doc_topic_matrix)
    max_weights = doc_topic_matrix.max(axis=1)
    doc_topics = doc_topic_matrix.argmax(axis=1)

    # Assign -1 to documents with weights below the minimum confidence threshold
    min_confidence = params.get("min_confidence", 0.00)
    doc_topics[max_weights <= min_confidence] = -1

    # Sort topics and reassign document-topic labels from biggest to smallest topic by document count
    doc_topics, topics = sort_topic_result(doc_topics, raw_topics, outlier=-1)
    log.info("Max topic probability for first 5 documents: %s", np.max(doc_topic_matrix, axis=1)[:5])
    log.info("Document-topic assignments shape: %s", doc_topics.shape)
    return TopicResult(doc_topics, topics, {"model": model, "embeddings": emb})
 


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODELS: dict[str, Callable[[list[str], dict, int], TopicResult]] = {
    "nmf": fit_nmf,
    "lda": fit_lda,
    "bertopic": fit_bertopic,
    "keynmf": fit_keynmf,
}


def get_model(name: str) -> Callable[[list[str], dict, int], TopicResult]:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODELS)}")
    return MODELS[name]