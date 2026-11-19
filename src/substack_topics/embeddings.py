from pathlib import Path
from dotenv import load_dotenv
from pyprojroot import here
import logging

import numpy as np
import hashlib

load_dotenv(here() / ".env")
log = logging.getLogger(__name__)


def get_embeddings(texts: list[str], 
                   model_name: str = "all-MiniLM-L6-v2",
                   cache_dir: Path | str = "data/interim/embeddings", # make sure its from root
                   batch_size: int = 128) -> np.ndarray:
    """Sentence embeddings with an on-disk cache keyed by model name + the exact texts.

    Args:
        texts: List of input texts to embed.
        model_name: Name of the sentence transformer model. Defaults to "all-MiniLM-L6-v2".
        cache_dir: Directory to store cached embeddings. Defaults to "data/interim/embeddings".
        batch_size: Batch size for embedding computation.

    Returns:
        Numpy array of shape (len(texts), embedding_dim) containing the embeddings.
    """
    # Create a unique hash for the combination of model name and texts to use as cache key
    h = hashlib.md5(model_name.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))

    path = Path(here(cache_dir)) / f"{model_name.replace('/', '_')}_{h.hexdigest()[:12]}.npy"
    if path.exists():
        log.info(f"Loading embeddings from cache: {path.relative_to(here())}")
        return np.load(path)

    # Generate embeddings using the specified model
    from sentence_transformers import SentenceTransformer
    log.info(f"Generating embeddings using model: {model_name}")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    log.info(f"Saving embeddings to cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)

    return emb
