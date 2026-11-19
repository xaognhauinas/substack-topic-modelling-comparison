import pandas as pd
from typing import Sequence
from functools import lru_cache

import html
import fasttext
import unicodedata
import urllib.request
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from pathlib import Path
import logging
import warnings
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


LOW_INFORMATION_TEXT_PATTERN = "|".join([
    r"\bcoming soon\b", r"^join my\b", r"^join me\b", r"^join us\b", 
])

LOW_INFORMATION_URL_SLUG_PATTERN = "|".join([
    r"\bcoming-soon\b", r"\bjoin-my\b", r"\bjoin-me\b", r"\bwelcome-to-my-substack\b",
    r"^welcome$", r"^hello$", 
])

def fill_missing_text(texts: pd.Series) -> pd.Series:
    """Fills missing data values with empty strings and strips outer whitespace."""
    return texts.fillna("").astype(str).str.strip()

def _format_html_string(text: str) -> str:
    """Extracts raw text from an individual string, handling HTML tags and encodings dynamically."""
    
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        raw_text = soup.get_text(separator=" ", strip=True)
    else:
        raw_text = text
        
    return html.unescape(raw_text).replace("\n", " ")

def clean_and_unescape_html(texts: pd.Series) -> pd.Series:
    """Strips all HTML tags and unescapes entity symbols in a single high-performance step."""
    
    processed_list = [_format_html_string(text) for text in fill_missing_text(texts)]

    return pd.Series(processed_list, index=texts.index).str.strip()

def build_text(titles: pd.Series, subtitles: pd.Series) -> pd.Series:
    """Vectorized concatenation of title and subtitle series."""
    titles = fill_missing_text(titles)
    subtitles = fill_missing_text(subtitles)
    return pd.Series(
        [f"{t}. {s}" if s else t for t, s in zip(titles, subtitles)],
        index=titles.index
    )

def normalise_series_title(text: pd.Series) -> pd.Series:
    """Transforms things like: 'Construction Curiosities #133' -> 'construction curiosities 0'
        
    Transforms text to lowercase, replaces all numbers with a static '0', strips 
    out punctuation, and cleans up double spaces. Useful for matching running 
    newsletter editions, numbered parts, or recurring text templates.
    """
    
    t = fill_missing_text(text).str.lower()
    t = t.str.replace(r"\d+", "0", regex=True)        # Turn all numbers into '0'
    t = t.str.replace(r"[^\w\s]", " ", regex=True)    # Replace all punctuation with spaces
    t = t.str.replace(r"\s+", " ", regex=True)        # Collapse multiple spaces into a single space
    
    return t.str.strip()

def url_slug(urls: pd.Series) -> pd.Series:
    """The part after /p/ in a Substack post URL."""
    return urls.astype(str).str.extract(r"/p/([^/?#]+)", expand=False)

def match_pattern(text: pd.Series, pattern: str) -> pd.Series:
    return text.str.contains(pattern, case=False, regex=True, na=False)


def flag_low_information_posts(
    text_series: pd.Series, 
    url_series: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generates a unified boolean mask identifying announcement posts 
    based on text patterns, URL slug patterns, and missing URLs.
    """
    slug_series = url_slug(url_series)

    kw_mask = match_pattern(text_series, LOW_INFORMATION_TEXT_PATTERN)
    slug_mask = match_pattern(slug_series, LOW_INFORMATION_URL_SLUG_PATTERN)
    empty_slug_mask = slug_series.isna()
    combined_mask = kw_mask | slug_mask | empty_slug_mask

    return combined_mask, kw_mask, slug_mask, empty_slug_mask



def calculate_text_structural_metrics(series: pd.Series) -> pd.DataFrame:
    """Calculates lengths, token counts, and basic text volume density profiles."""
    metrics = pd.DataFrame(index=series.index)
    
    # 1. Core measurements using native Pandas vectorized strings
    char_len = series.str.len()
    letter_count = series.str.count(r"[^\W\d_]")
    
    # 2. Derive densities and token allocations
    metrics["char_length"] = char_len
    metrics["letter_length"] = letter_count
    metrics["letter_ratio"] = letter_count / char_len.clip(lower=1)
    metrics["is_letter_poor"] = series.ne("") & (metrics["letter_ratio"] < 0.5)
    metrics["word_count"] = series.str.split().str.len().fillna(0).astype(int)
    
    return metrics

@lru_cache(maxsize=None)
def _char_script(ch: str) -> str:
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return "UNKNOWN"

def dominant_script(text: str) -> str:
    """Most common Unicode script among the letters: LATIN, CJK, CYRILLIC, ARABIC, HANGUL, ..."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            sc = _char_script(ch)
            counts[sc] = counts.get(sc, 0) + 1
    return max(counts, key=counts.get) if counts else "NONE"

def dominant_script_by_word(text: str) -> str:
    """Finds the dominant script by counting words instead of individual letters."""
    counts: dict[str, int] = {}
    
    for word in text.split():
        # Clean the word to only check letters
        letters = [ch for ch in word if ch.isalpha()]
        if not letters:
            continue
        
        # Get the script of the first letter of the word
        word_script = _char_script(letters[0])
        counts[word_script] = counts.get(word_script, 0) + 1
        
    return max(counts, key=counts.get) if counts else "NONE"

def ensure_fasttext_model(model_path: str | Path) -> str:
    """Ensure the fastText model file exists at the given path.

    Resolves the path to an absolute path, creates missing directories, 
    and automatically downloads the model file if it is missing.
    
    Returns:
        str: The absolute string path to the validated model file.
    """
    # 1. Safely cast to a Path object and resolve to a clean absolute path immediately
    resolved_path = Path(model_path).resolve()

    # 2. If the target file is missing, prepare directories and download it
    if not resolved_path.exists():
        target_directory = resolved_path.parent
        target_directory.mkdir(parents=True, exist_ok=True)

        print(f"Downloading model to exact absolute path: {resolved_path}...")
        model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        
        try:
            urllib.request.urlretrieve(model_url, str(resolved_path))
            print("Download complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to download fastText model from {model_url}. Error: {e}")
        
    return str(resolved_path)


def detect_languages(texts: pd.Series, fasttext_model_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame with language codes and confidence scores per text row.

    Automatically handles directory creation and model downloading if the 
    specified fasttext_model path does not exist on disk.
    """
    # Verify, resolve, and download the model file path instantly before parsing data
    model_string_path = ensure_fasttext_model(fasttext_model_path)
    # 1. Pre-process texts: clean text, unescape HTML entities, and strip newlines
    clean_texts = clean_and_unescape_html(texts)
    # 2. Determine the dominant script for each cleaned text row
    script_series = clean_texts.map(dominant_script_by_word)
    # 3. Load the pre-trained fastText language identification model
    ft = fasttext.load_model(model_string_path)
    
    
    try:
        labels, scores = ft.predict(clean_texts.tolist(), k=1)
        
        lang_codes = [lbl[0].replace("__label__", "") for lbl in labels]
        confidences = [float(scr[0]) for scr in scores]
        
    except Exception as e:
        raise RuntimeError("fastText batch prediction failed") from e

    # Return structured DataFrame matching the index of the original texts
    return pd.DataFrame({
        "text": texts,
        "clean_text": clean_texts,
        "script": script_series,
        "language": lang_codes,
        "confidence": confidences
    }, index=texts.index)


def _log_dropped_metrics(previous_total: int, current_total: int, step_name: str) -> dict:
    """Log one pipeline step and return it as a row of the funnel table."""

    dropped = previous_total - current_total
    drop_pct = 100 * dropped / previous_total if previous_total else 0.0
    log.info(f"[{step_name}] Dropped {dropped:,} rows ({drop_pct:.2f}%) | Remaining {current_total:,} rows ({100 - drop_pct:.2f}%)")

    return {"step": step_name, "remaining": current_total, "dropped": dropped, "dropped_pct_of_previous": round(drop_pct, 3)}

def preprocess_pipeline(
    df: pd.DataFrame, 
    min_date: str,
    max_date: str,
    fasttext_model_path: str,
    # url_col: str = "article_url", 
    date_col: str = "publish_date", 
    title_col: str = "post_title", 
    sub_col: str = "subtitle_summary",
    select_cols: Sequence[str] = ("article_url", "publication_domain", "author_name", "category",
                                   "post_title", "subtitle_summary", "publish_date"),
    return_funnel: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates the data engineering pipeline for text sanitation and filtering.
 
    Performs a series of preprocessing steps on the input DataFrame, including:
    1. Date filtering
    2. HTML cleaning and text combination
    3. Language and script detection with fastText
    4. Deduplication based on series keys
    5. Low-information post filtering (announcement and recruitment)
    6. Text quality filtering
    
    Returns a cleaned and filtered DataFrame ready for downstream processing. 
    With return_funnel=True, it returns (cleaned_df, funnel_df), where funnel_df has one row per step (remaining, dropped, % dropped).
    """
    initial_total = len(df)
    output_df = df[list(select_cols)].copy()
    funnel = [{"step": "Raw dataset", "remaining": initial_total, "dropped": 0, "dropped_pct_of_previous": 0.0}]
    
    # -------------------------------------------------------------------------
    # Step 1: Initial Date Pruning
    # -------------------------------------------------------------------------
    output_df = output_df[(output_df[date_col] >= min_date) & (output_df[date_col] <= max_date)]
    output_df["publish_year"] = output_df["publish_date"].dt.year
    
    log.info(f"Initial date filter applied: {min_date} <= post_date <= {max_date}")
    funnel.append(_log_dropped_metrics(initial_total, len(output_df), "Date Filter"))

    # -------------------------------------------------------------------------
    # Step 2: HTML Text Sanitation (Executed before language detection)
    # -------------------------------------------------------------------------
    # Strips layout tags and decodes entities 
    output_df[title_col] = clean_and_unescape_html(output_df[title_col])
    output_df[sub_col] = clean_and_unescape_html(output_df[sub_col])    
    output_df["text_col"] = build_text(output_df[title_col], output_df[sub_col])
 
    # -------------------------------------------------------------------------
    # Step 3: High-Accuracy Language & Script Pruning
    # -------------------------------------------------------------------------
    current_total = len(output_df)
    
    text_lang_df = detect_languages(output_df["text_col"], fasttext_model_path=fasttext_model_path)
    
    # Language mask: English language above 0.10 threshold and Latin or Mathematical script targets
    lang_mask = (
        (text_lang_df["language"] == "en") & 
        (text_lang_df["script"].isin(["LATIN", "MATHEMATICAL"])) & 
        (text_lang_df["confidence"] > 0.10)
    )
    output_df = output_df[output_df.index.isin(text_lang_df[lang_mask].index)]
    funnel.append(_log_dropped_metrics(current_total, len(output_df), "Language & Script Filter"))

    # -------------------------------------------------------------------------
    # Step 4: Deduplicate Recurring Series Entries
    # -------------------------------------------------------------------------
    current_total = len(output_df)
    output_df["series_key"] = normalise_series_title(output_df["text_col"])

    output_df = output_df.drop_duplicates(subset=["publication_domain", "series_key"])
    funnel.append(_log_dropped_metrics(current_total, len(output_df), "Series Deduplication"))

    # -------------------------------------------------------------------------
    # Step 5: Identify and Drop Low-Information Posts (Announcement and Recruitment)
    # -------------------------------------------------------------------------
    current_total = len(output_df)

    # Flag announcement signatures and recruitment notices in the text column
    low_info_mask = flag_low_information_posts(output_df["text_col"],output_df["article_url"])[0]
    output_df = output_df[~low_info_mask]
    funnel.append(_log_dropped_metrics(current_total, len(output_df), "Low-Information Post (Announcement and Recruitment) Filtering"))

    # -------------------------------------------------------------------------
    # Step 6: Text Quality Filtering
    # -------------------------------------------------------------------------
    current_total = len(output_df)
    text_quality_df = calculate_text_structural_metrics(output_df["text_col"])

    # Filter out posts with poor letter composition or very short word count
    text_quality_df = text_quality_df[(text_quality_df["is_letter_poor"] == False) & (text_quality_df["word_count"] > 3)]
    output_df = output_df[output_df.index.isin(text_quality_df.index)]
    funnel.append(_log_dropped_metrics(current_total, len(output_df), "Text Quality Filtering"))


    # Final reporting hook
    log.info(f"Pipeline Complete. Retained {len(output_df)} ({len(output_df)/initial_total:.2%}) out of {initial_total} original rows.")
    if return_funnel:
        funnel_df = pd.DataFrame(funnel)
        funnel_df["remaining_pct_of_raw"] = (100 * funnel_df["remaining"] / initial_total).round(2)
        return output_df, funnel_df
    return output_df

