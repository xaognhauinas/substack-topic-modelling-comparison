import numpy as np
import pandas as pd
from IPython.display import display

from .clean import *

# Helpers -----------------------------------------------------------------------------------------
UNSAFE_COLS = ["author_name"]

def safe_view(
    df: pd.DataFrame,
    n: int = 10,
    cols: list[str] | None = None,
    mask=None,
    seed: int = 0,
) -> pd.DataFrame:
    """Random sample of rows WITHOUT author names. Use this for all displays."""
    # Remove unsafe columns if cols is None
    chosen_cols = cols if cols is not None else [c for c in df.columns if c not in UNSAFE_COLS]

    # Apply mask, filter down to existing columns, and sample safely
    d = df[mask] if mask is not None else df
    valid_cols = [c for c in chosen_cols if c in d.columns]

    if d.empty:
        return d[valid_cols]

    return d[valid_cols].sample(min(n, len(d)), random_state=seed)

def calculate_percentage(n: float, total: float) -> float:
    return round(100 * n / total, 3) if total else float("nan")

def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)

def _table(
    rows: list[tuple[str, float]], total: int, name: str = "rows"
) -> pd.DataFrame:
    """Create a summary table with counts and shares for a list of metric rows."""
    data = [(m, int(n), calculate_percentage(n, total)) for m, n in rows]
    return pd.DataFrame(data, columns=["metric", name, "pct_share"])


# Text: Length & Quality --------------------------------------------------------------------------------------
# Raw HTML character references that didnt get decoded into text
# Checks for 3 separate formatting variations 
# (decimal numeric entities `&#10;`, hexadecimal numeric entities `&#x2014;`, named entities between 2 and 8 letters long `&amp;`)
ENTITY_RE = r"&#\d+;|&#x[0-9a-fA-F]+;|&[a-zA-Z]{2,8};"
# Character encoding failure: looks for garbage characters caused by mismatched text decoding
MOJIBAKE_RE = (
    r"Ã[¢-¥©®°±²³´¶·¹º»¼½¾¿]"          # Strict character set
    r"|â€[”™˜œ\x9d]|â€”|â€“"            # Targets common smart punctuation fails
    r"|Â[£€¢¥©®°±²³´¶·¹º»¼½¾¿]"         # Catches standard symbol prefix errors
    r"|" + "\u201c\u201d\ufffd"         # Directly appends the true Unicode characters
)
# Emojis and Symbols
EMOJI_RE = "[\U0001F300-\U0001FAFF\u2600-\u27BF]"


def _detect_pattern_flags(series: pd.Series) -> pd.DataFrame:
    """Scans for formatting issues like HTML tags, broken encodings, or emojis."""
    flags = pd.DataFrame(index=series.index)
    
    flags["has_html"] = match_pattern(series, ENTITY_RE)
    flags["has_mojibake"] = match_pattern(series, MOJIBAKE_RE)

    # Decode HTML character references before further pattern matching
    decoded_series = clean_and_unescape_html(series)
    flags["has_emoji"] = match_pattern(decoded_series, EMOJI_RE)
    
    return flags, decoded_series


def profile_text_column(series: pd.Series) -> pd.DataFrame:
    """Generates a complete quality and structural analysis DataFrame for one column.
    
    Safely sanitizes text input values before executing analysis sweeps.
    """

    clean_series = fill_missing_text(series)

    pattern_df, decoded_series = _detect_pattern_flags(clean_series)
    structural_df = calculate_text_structural_metrics(decoded_series)
    
    # Combine individual results side-by-side
    return pd.concat([decoded_series.rename(f"{series.name}_decoded"), structural_df, pattern_df], axis=1)


def text_quality_summary(
    df: pd.DataFrame,
    title_col: str = "post_title", 
    sub_col: str = "subtitle_summary",
) -> pd.DataFrame:
    """Returns a standalone summary DataFrame containing targeted text fields and quality metrics.

    """
    # Start with a clean copy containing only raw target columns
    summary_df = df[[title_col, sub_col]].copy()
    summary_df["combined_text"] = build_text(df[title_col], df[sub_col])

    # Accumulator to hold profile blocks before merging back
    profile_blocks = []

    for col in summary_df.columns:
        # Generate the metrics block for this specific text field
        col_profile = profile_text_column(summary_df[col])
        
        # Add the prefix dynamically to track metric sources (e.g., 'post_title_word_count')
        col_profile.columns = [f"{col}_{metric}" for metric in col_profile.columns]
        profile_blocks.append(col_profile)
        
    # Stitch everything back together side-by-side instantly using Pandas axis=1 matching
    return pd.concat([summary_df] + profile_blocks, axis=1)


# Text: Duplicates, Recurring Titles, Series --------------------------------------------------------------------------
def display_duplicate_breakdown(
        df: pd.DataFrame, 
        top_n: int = 5,    
        url_col: str = "article_url", 
        title_col: str = "post_title", 
        sub_col: str = "subtitle_summary",
        ) -> pd.DataFrame:
    """Loops through each duplication metric and displays its top offenders as isolated DataFrames."""
    
    text_df = df.copy()[["publication_domain", title_col, sub_col, url_col]]
    
    # Base cleaning for exact matching
    text_df["clean_title"] = clean_and_unescape_html(fill_missing_text(text_df[title_col]))
    text_df["clean_sub"] = clean_and_unescape_html(fill_missing_text(text_df[sub_col]))
    
    # Normalized skeleton keys for structural matching
    text_df["series_title_key"] = normalise_series_title(text_df["clean_title"])
    text_df["series_sub_key"] = normalise_series_title(text_df["clean_sub"])
    
    # Set up clean masks on the normalized series keys to filter out empty strings
    nonempty_title_key = text_df["series_title_key"].ne("")
    nonempty_sub_key = text_df["series_sub_key"].ne("")

    # Extract URL slugs for later analysis
    text_df["url_slug"] = url_slug(text_df[url_col])
    
    # Define tasks—routing exact matches to clean_* and series groupings to series_*_key
    configs = [
        ("1. Duplicate Article URLs", [url_col], None),
        ("2. Exact Clones (Publication + Exact Title + Exact Subtitle)", ["publication_domain", "clean_title", "clean_sub"], None),
        ("3. Exact Clones After Taking Series into Account (Publication + Exact Title + Exact Subtitle)", ["publication_domain", "series_title_key", "series_sub_key"], None),
        ("4. Recurring Publication Titles (Same Titles within a Publication)", ["publication_domain", "clean_title"], nonempty_title_key),
        ("5. Recurring Publication Titles After Taking Series into Account (Same Titles within a Publication)", ["publication_domain", "series_title_key"], nonempty_title_key),
        ("6. Shared Title + Subtitle Patterns (Across Different Publications)", ["clean_title", "clean_sub"], nonempty_title_key & nonempty_sub_key),
        ("7. Duplicated Titles Globally", ["clean_title"], nonempty_title_key),
        ("8. Duplicated Subtitles Globally", ["clean_sub"], nonempty_sub_key),
        ("9. Duplicated URL Slugs Globally", ["url_slug"], None),
    ]
    
    # 4. Loop through configurations, display individual grids, and track summaries safely
    metrics = []
    for title_label, group_cols, mask in configs:
        working_df = text_df if mask is None else text_df[mask]
        
        if working_df.empty:
            metrics.append((title_label, 0))
            print(f"\n--- {title_label} ---\nNo records found matching this criteria.")
            continue
            
        # Aggregate occurrences for high-frequency printing
        cluster_df = (
            working_df.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="Total Count")
            .query("`Total Count` > 1")
            .sort_values(by="Total Count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        
        # Calculate overall true duplicate counts for the report card
        total_dups = int(working_df.duplicated(subset=group_cols).sum())
        metrics.append((title_label, total_dups))
        
        # Raw printing per loop iteration
        print(f"\n--- {title_label} ---")
        if cluster_df.empty:
            print("No duplicates found in this category.")
        else:
            display(cluster_df)

    return _table(metrics, len(df))


# Text: Language ----------------------------------------------------------------------------------------
def language_summary(
    fasttext_model_path: str,
    df: pd.DataFrame,
    title_col: str = "post_title", 
    sub_col: str = "subtitle_summary",
) -> pd.DataFrame:
    """Generate summary analysis for scripts and languages by combining context columns.

    Safely handles missing data and avoids modifying the input DataFrame.
    """
    total_rows = len(df)
    
    combined_text = build_text(df[title_col], df[sub_col])
    
    # Extract scripts and languages on the high-context combined data
    lang_df = detect_languages(combined_text, fasttext_model_path=fasttext_model_path)

    print("\n--- Confidence Score Distribution ---")
    display(lang_df['confidence'].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

    # Profile Script Shares
    print("\n--- Script Shares Profile ---")
    script_summary_df = (
        lang_df.groupby("script")
        .size()
        .reset_index(name="size")
        .sort_values(by="size", ascending=False)
        .assign(pct_share=lambda x: calculate_percentage(x["size"], total_rows))
    )
    display(script_summary_df.head(20))

    # Profile Language Shares
    print("\n--- Language Shares Profile ---")
    lang_summary_df = (
        lang_df.groupby("language")
        .size()
        .reset_index(name="size")
        .sort_values(by="size", ascending=False)
        .assign(pct_share=lambda x: calculate_percentage(x["size"], total_rows))
    )
    display(lang_summary_df.head(20))


    # Catch Structural Anomalies (e.g., English text using non-LATIN character sets)
    print("\n--- Alignment Metrics & Anomalies ---")
    mismatched_df = lang_df[(~lang_df["script"].eq("LATIN")) & (lang_df["language"].eq("en"))]
    mismatch_pct = calculate_percentage(len(mismatched_df), total_rows)
    print(f"Mismatch - English language but not LATIN script: {len(mismatched_df)} rows, {mismatch_pct}%")
    display(mismatched_df.head(20))

    # Profile Clean Filter Targets
    filtered_df = lang_df[(lang_df["script"].isin(["LATIN", "MATHEMATICAL"]) & lang_df["language"].eq("en"))]
    filtered_pct = calculate_percentage(len(filtered_df), total_rows)
    print(f"Filtered LATIN/MATHEMATICAL English content: {len(filtered_df)} rows, {filtered_pct}%")

    filtered_df = filtered_df[filtered_df['confidence'] > 0.1]
    filtered_pct = calculate_percentage(len(filtered_df), total_rows)
    print(f"Filtered LATIN/MATHEMATICAL English content with confidence > 0.1: {len(filtered_df)} rows, {filtered_pct}%")


    return lang_df

# Text: Low Information Posts ---------------------------------------------------------------------------- 
def low_information_posts_summary(
        df: pd.DataFrame, 
        url_col: str = "article_url", 
        title_col: str = "post_title", 
        sub_col: str = "subtitle_summary", 
        sample: int = 25, 
        seed: int = 0) -> dict:
    """Generates a diagnostic report on announcement-style posts using text patterns and URL placeholders"""
    text = build_text(df[title_col], df[sub_col])

    combined_mask, kw_mask, slug_mask, empty_slug_mask = flag_low_information_posts(text, df[url_col])
    
    summary = _table([
        ("keyword-flagged low-information posts (announcement and recruitment)", kw_mask.sum()),
        ("slug-flagged low-information posts (announcement and recruitment)", slug_mask.sum()),
        ("empty-slug posts", empty_slug_mask.sum()),
        ("either", combined_mask.sum()),
    ], len(df))
    return {
        "summary": summary, 
        "sample-keyword-filtered": safe_view(df, n=sample, mask=kw_mask, seed=seed),
        "sample-slug-filtered": safe_view(df, n=sample, mask=slug_mask, seed=seed),
        "sample-empty-slug": safe_view(df, n=sample, mask=empty_slug_mask, seed=seed),
    }
