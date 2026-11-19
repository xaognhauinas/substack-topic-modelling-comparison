from pathlib import Path
import pandas as pd

# Download and load functions for the Substack dataset
def download_kaggle_dataset(handle: str, output_dir: str | Path) -> Path:
    """Download a Kaggle dataset using kagglehub.

    Args:
        handle: The Kaggle dataset handle (e.g., 'user/dataset-name').
        output_dir: The root directory where kagglehub will cache the data.

    Returns:
        Path: The exact path where the downloaded files are located.
    """
    import kagglehub

    # Clean up the path and make it absolute
    cache_dir = Path(output_dir).resolve()

    # Download and convert the resulting string back into a Path object
    downloaded_path = kagglehub.dataset_download(handle, output_dir=str(cache_dir))

    return Path(downloaded_path)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from a single parquet, CSV, or JSONL file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(file_path, engine="fastparquet")

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in (".jsonl", ".json"):
        return pd.read_json(file_path, lines=True)

    raise ValueError(f"Unsupported file type: {file_path}")

