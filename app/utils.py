"""Shared helpers for the Streamlit app.

The app only reads pre-computed files from `artefacts/` (small CSV and JSON summaries exported by the notebooks)
and never fits models at runtime, keeping deployment dependencies minimal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from dataclasses import dataclass
import streamlit as st

ARTEFACTS = Path(__file__).resolve().parent.parent / "artefacts"

# Same colours as the notebook charts, so the app and the notebooks look like one project.
COLOR_PALETTE = ["#83241E", "#102020", "#78B4B7","#2D4C4F", "#D9E9E9"]
MODEL_COLOURS = {"nmf": COLOR_PALETTE[1], 
                 "lda": COLOR_PALETTE[2], 
                 "bertopic": COLOR_PALETTE[3], 
                 "keynmf": COLOR_PALETTE[4]}
MODEL_NAMES = {"nmf": "1. TF-IDF + NMF", 
               "lda": "2. LDA (gensim)", 
               "bertopic": "3. BERTopic", 
               "keynmf": "4. KeyNMF (Turftopic)"}
MODEL_ORDER = ["nmf", "lda", "bertopic", "keynmf"]
PLOT_TEMPLATE = "plotly_white"


@st.cache_data(show_spinner=False)
def load_csv(folder_name: str, file_name: str) -> pd.DataFrame | None:
    path = ARTEFACTS / folder_name / file_name
    if not path.exists():
        raise FileNotFoundError(f"artefacts/{folder_name}/{file_name}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(folder_name: str, file_name: str):

    path = ARTEFACTS / folder_name / file_name
    if not path.exists():
        raise FileNotFoundError(f"artefacts/{folder_name}/{file_name}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_artefact(loader, folder_name:str, file_name: str):
    """Load an artefact or stop the page politely if missing."""
    try:
        return loader(folder_name, file_name)
    except FileNotFoundError as missing:
        st.warning(f"`{missing}` is missing. Run the export cell at the end of the matching notebook.")
        st.stop()

CALLOUT_CONFIG = {
    "observation": ("Observation", ":material/visibility:"),
    "decision": ("Decision", ":material/check_circle:"),
    "takeaway": ("Key Takeaway", ":material/lightbulb:"),
}

def callout(kind: str, text: str) -> None:
    label, icon = CALLOUT_CONFIG[kind]
    st.markdown(f":gray-badge[{icon} {label}] {text}")

def callouts(kind: str, *texts: str) -> None:
    """Several notes of the same kind, one line each."""
    for text in texts:
        callout(kind, text)

def format_int(n) -> str:
    return f"{int(n):,}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%"


# Evaluation Metrics ------------------------------------------------------------
@dataclass(frozen=True)
class MetricMeta:
  label: str
  short_name: str
  question: str
  theoretical_range: str
  typical_range: str
  interpretation: str
  lower_is_better: bool = False
  fmt: str = "%.3f"
  scale: float = 1.0

  @property
  def help_text(self) -> str:
    """Formats metadata into clean Markdown for Streamlit's help tooltip."""
    return (
        f"**{self.label}**\n\n"
        f"*{self.question}*\n\n"
        f"- **Theoretical range:** `{self.theoretical_range}`\n"
        f"- **Typical range:** `{self.typical_range}`\n\n"
        f"{self.interpretation}"
    )


KEY_METRICS: dict[str, MetricMeta] = {
    "coherence_c_npmi": MetricMeta(
        label="Co-occurrence coherence (c_npmi)",
        short_name="Coherence  \n(co-occurrence)",
        question="Do a topic's top words appear together in the same posts?",
        theoretical_range="-1 to +1",
        typical_range="-0.10 to +0.35",
        interpretation=(
            "Higher is better. +1 means words always appear together, 0 means"
            " random chance, and -1 means they never appear together. Short"
            " texts like titles typically score lower (around 0.05 to 0.25)."
        ),
    ),
    "wec": MetricMeta(
        label="Word-embedding coherence (WEC)",
        short_name="Coherence (word-embedding)",
        question=(
            "Do a topic's top words have similar meanings in a word-vector"
            " space?"
        ),
        theoretical_range="-1 to +1",
        typical_range="0.10 to 0.65",
        interpretation=(
            "Higher is better. Measures average cosine similarity between a"
            " topic's top word vectors. Although the scale runs from -1 to +1,"
            " pretrained vectors (such as GloVe) rarely drop below 0 for top"
            " words."
        ),
    ),
    "centroid_sim_diff": MetricMeta(
        label="Document tightness above chance",
        short_name="Tightness",
        question=(
            "Are a topic's posts closer to their topic centre than random"
            " posts?"
        ),
        theoretical_range="-2 to +2",
        typical_range="0.00 to +0.40",
        interpretation=(
            "Higher is better. Measures how much closer posts are to their own"
            " topic centre compared with a random baseline. 0 means no tighter"
            " than random, while negative values point to scattered topics."
        ),
    ),
    "topic_diversity": MetricMeta(
        label="Topic diversity",
        short_name="Diversity",
        question="Do different topics use distinct sets of words?",
        theoretical_range="0 to 1",
        typical_range="0.50 to 1.00",
        interpretation=(
            "Higher is better. Share of unique top words across all topics. A"
            " score of 1.0 means no words are repeated between topics."
        ),
    ),
    "outlier_rate": MetricMeta(
        label="Outlier rate (%)",
        short_name="Outliers",
        question="What share of posts is left unassigned?",
        theoretical_range="0% to 100%",
        typical_range="5% to 50%",
        interpretation=(
            "Lower is better, though 0% can mean forced assignment. Share of"
            " posts labelled -1. Values above 50% usually mean the clustering"
            " settings are too strict."
        ),
        lower_is_better=True,
        fmt="%.1f%%",
        scale=100.0,
    ),
    "largest_topic_relative_%": MetricMeta(
        label="Largest topic (% of assigned posts)",
        short_name="Largest topic (% of assigned posts)",
        question="What share of assigned posts does the largest topic hold?",
        theoretical_range="0% to 100%",
        typical_range="5% to 40%",
        interpretation=(
            "Lower is better. Share of assigned posts placed into the single"
            " biggest topic. Values above 40% to 50% usually point to a"
            " catch-all topic absorbing unrelated posts."
        ),
        lower_is_better=True,
        fmt="%.1f%%",
    ),
    "largest_topic_absolute_%": MetricMeta(
        label="Largest topic (% of all posts)",
        short_name="Largest topic (% of all posts)",
        question="What share of all posts does the largest topic hold?",
        theoretical_range="0% to 100%",
        typical_range="5% to 40%",
        interpretation=(
            "Lower is better. Share of all posts placed into the single"
            " biggest topic. Values above 40% to 50% usually point to a"
            " catch-all topic absorbing unrelated posts."
        ),
        lower_is_better=True,
        fmt="%.1f%%",
    ),
    "runtime_m": MetricMeta(
        label="Runtime (minutes)",
        short_name="Runtime",
        question="How long did the model take to fit?",
        theoretical_range="0 to +∞",
        typical_range="0.1 to 60+ min",
        interpretation=(
            "Lower is better. Total fitting time in minutes, depending on"
            " corpus size and encoder model size (note that embeddings are"
            " cached)."
        ),
        lower_is_better=True,
        fmt="%.1f min",
    ),
}

LOWER_METRIC_IS_BETTER = {
    name
    for k, v in KEY_METRICS.items()
    if v.lower_is_better
    for name in (k, v.label, v.short_name)
}


def render_metric(col, m: pd.Series, key: str, label: str | None = None):
    """Renders a single st.metric card driven by KEY_METRICS metadata."""
    meta = KEY_METRICS[key]
    val = float(m[key]) * meta.scale
    formatted_val = meta.fmt % val

    col.metric(
        label or meta.short_name,
        formatted_val,
        help=meta.help_text,
    )


def highlight_extremes(s: pd.Series, gap_ratio: float = 0.15) -> list[str]:
  """Highlights the highest value(s) in red if lower is better, else green."""
  clean = s.dropna()
  ordered = clean.nlargest(2)

  if len(ordered) < 2:
    winners = ordered.values
  else:
    col_span = clean.max() - clean.min()
    gap = abs(ordered.iloc[0] - ordered.iloc[1])
    is_far_apart = (col_span > 0) and (gap / col_span > gap_ratio)
    winners = ordered.iloc[:1].values if is_far_apart else ordered.values

  color = "#ff4b4b" if s.name in LOWER_METRIC_IS_BETTER else "#21c354"
  style = f"color: {color}; font-weight: bold;"

  return [style if v in winners else "" for v in s]
