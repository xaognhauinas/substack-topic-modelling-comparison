"""Tokenisation helpers. Each model family needs different input:
    - Gensim LDA needs tokenised, cleaned text (this module).
    - BERTopic wants near-raw text (the embedding model uses context); stopwords are removed
    only in the vectoriser that labels topics.
    - Coherence is computed on ONE shared tokenisation (simple_tokenize) for every model.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from functools import lru_cache

# Regular expression for tokenisation: letters only, 2 or more characters
_TOKEN_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)  # letters only, 2+ chars
STOP_WORDS_SET = set(ENGLISH_STOP_WORDS)

@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])

def simple_tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS_SET]

def spacy_tokeniser(texts: list[str], n_process: int = 1) -> list[list[str]]:
    """Tokenises texts using a pre-loaded spaCy model configuration."""
    tokens = []
    
    # Lemmatise and filter tokens (punctuation, stop words and non-relevant parts of speech)
    for doc in _nlp().pipe(texts, batch_size=1000, n_process=n_process):
        tokens.append([
            t.lemma_.lower() for t in doc
            if t.pos_ in {"NOUN", "PROPN", "ADJ"} and t.is_alpha and not t.is_stop and len(t) > 1
        ])
    return tokens

def tokenise_for_lda(
    texts: list[str], 
    add_bigrams: bool = False, 
    bigram_min_count: int = 10, 
    bigram_threshold: float = 10.0,
) -> list[list[str]]:
    """ Tokenise for LDA using spaCy lemmas (nouns, proper nouns, and adjectives). 
    Optionally combines frequent word pairs into bigrams.
    """
    # Run tokeniser
    tokens = spacy_tokeniser(texts)
    
    # Add bigrams if requested
    if add_bigrams:
        from gensim.models import Phrases
        
        # Train the phrase detector on your tokenized text
        phrase_model = Phrases(tokens, min_count=bigram_min_count, threshold=bigram_threshold)
        
        # Export to a faster, lighter Phraser object for transformation
        bigram_transformer = phrase_model.freeze()
        
        # Apply the bigrams (e.g., transforming ['machine', 'learning'] -> ['machine_learning'])
        tokens = [bigram_transformer[doc_tokens] for doc_tokens in tokens]

    return tokens





