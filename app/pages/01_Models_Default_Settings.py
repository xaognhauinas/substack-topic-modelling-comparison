"""Notebook 01: Baseline evaluation of four topic models at default settings on the same data, seed, and metrics."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (KEY_METRICS, LOWER_METRIC_IS_BETTER, MODEL_COLOURS, MODEL_NAMES, MODEL_ORDER, PLOT_TEMPLATE, 
                   callout, callouts, format_int,load_csv, load_json, get_artefact, render_metric, highlight_extremes)

folder_name = "01_models_default_settings"

st.set_page_config(page_title="Models at default settings", page_icon="🧪", layout="wide")

setup = get_artefact(load_json, folder_name, "model_setup.json")
metrics = get_artefact(load_csv, folder_name, "model_default_metrics.csv").set_index("model").loc[MODEL_ORDER]
cards = get_artefact(load_csv, folder_name, "model_default_topic_cards.csv")
N_DOCS = setup["modelled_posts"]
COLOURS = {MODEL_NAMES[k]: v for k, v in MODEL_COLOURS.items()}

st.title("Models at default settings")
st.caption(
    "Notebook 01: Evaluates four topic-modelling approaches, ranging from traditional to state-of-the-art, "
    "fitted at default settings on the same corpus and scored across identical metrics to compare baseline "
    "behaviour and failure modes prior to tuning."
)
with st.expander("How to read the notes in each tab"):
    callout("observation", "The figures and patterns observed in the results.")
    callout("takeaway", "What these observations mean for the model and how they inform tuning.")

tabs = st.tabs(["Setup", "TF-IDF + NMF", "LDA", "BERTopic", "KeyNMF", "Side by side", "Takeaways"])

# Model descriptions and helper functions -------------------------------------------------
WHAT_IT_DOES = {
    "nmf": [
        "Builds a TF-IDF word matrix and splits it into 20 topics using non-negative matrix factorisation.",
        "Fast and based purely on word counts, serving as the baseline benchmark for the other models.",
    ],
    "lda": [
        "A probabilistic model assuming each post is a mix of topics and each topic is a distribution over words.",
        "Fitted on lemmatised nouns, proper nouns, and adjectives extracted with spaCy.",
    ],
    "bertopic": [
        "Encodes each post with a sentence encoder (`all-MiniLM-L6-v2`), reduces dimensions with UMAP, clusters with HDBSCAN, and labels topics using c-TF-IDF.",
        "Posts that do not fit any cluster are assigned to the outlier group (`-1`).",
        "Initially produced 439 clusters, which were merged down to 20 topics plus the outlier group.",
    ],
    "keynmf": [
        "Extracts key phrases for each post using the same sentence encoder (`all-MiniLM-L6-v2`), then factorises the resulting keyword matrix with NMF.",
    ],
}

HYPOTHESIS = {
    "nmf": [
        "Topics will group tightly around shared keywords, since NMF relies strictly on exact word counts.",
    ],
    "lda": [
        "Because LDA relies heavily on document length to capture word co-occurrence patterns, it may struggle on short titles and subtitles.",
    ],
    "bertopic": [
        "Will achieve the strongest semantic grouping, clustering related posts even when they use different vocabulary.",
        "However, density-based clustering may produce a high outlier rate on short, noisy text.",
    ],
    "keynmf": [
        "Expected to strike a balance between speed and semantic quality compared with BERTopic.",
    ],
}


def bullets(items):
    return "\n".join(f"- {i}" for i in items)


def size_chart(model: str) -> pd.DataFrame:
    d = cards[cards["model"] == model].sort_values("size", ascending=False).reset_index(drop=True)
    basis = st.radio(
        "Topic size as a share of",
        ["assigned posts only", "all posts"],
        horizontal=True,
        key=f"basis_{model}",
    )
    col = "absolute_share_%" if basis == "all posts" else "relative_share_%"
    d["topic label"] = [f"Topic {i}" for i in range(len(d))]
    fig = px.bar(
        d.iloc[::-1],
        x=col,
        y="topic label",
        orientation="h",
        template=PLOT_TEMPLATE,
        hover_data={"top_words": True, "size": ":,", "topic label": False, col: ":.1f"},
        labels={col: f"% of {basis}", "topic label": ""},
        text=col,
    )
    fig.update_traces(marker_color=MODEL_COLOURS[model], texttemplate="%{text:.1f}", textangle=0)
    fig.update_layout(height=620, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    return d

def topic_table(model: str, d: pd.DataFrame) -> None:
    n = st.slider("Number of topics to display (largest first)", 3, len(d), min(10, len(d)), key=f"n_{model}")
    show = d.head(n)[["size", "relative_share_%", "absolute_share_%", "top_words", "examples"]].rename(
        columns={
            "size": "Posts", 
            "relative_share_%": "% of assigned posts", 
            "absolute_share_%": "% of all posts", 
            "top_words": "Top words",
            "examples": "Example posts",
            }
    )
    show["% of all posts"] = show["% of all posts"].round(1)
    st.dataframe(show, 
                 row_height=100
                 )
    st.caption(
        "*Random sample of posts per topic (separated by `||`), truncated for display.*"
    )


def model_tab(model: str):
    """Overview, headline metrics, topic size chart, and topic table. Returns the metrics row and topic table."""
    m = metrics.loc[model]
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**How it works**")
            st.markdown(bullets(WHAT_IT_DOES[model]))
    with right:
        with st.container(border=True):
            st.markdown("**Initial hypothesis**")
            st.markdown(bullets(HYPOTHESIS[model]))
            
    st.subheader("Results")
    col_qual, col_cov = st.columns([3, 4])

    with col_qual:
      with st.container(border=True):
        st.caption("**Topic Quality & Coherence**")
        q1, q2, q3 = st.columns(3)
        render_metric(q1, m, "coherence_c_npmi")
        render_metric(q2, m, "wec")
        render_metric(q3, m, "centroid_sim_diff")

    with col_cov:
      with st.container(border=True):
        st.caption("**Coverage & Efficiency**")
        c1, c2, c3, c4 = st.columns(4)
        render_metric(c1, m, "topic_diversity")
        render_metric(c2, m, "outlier_rate")
        render_metric(c3, m, "largest_topic_relative_%", label="Largest topic")
        render_metric(c4, m, "runtime_m")

    d = size_chart(model)
    topic_table(model, d)
    st.subheader("Observations and Takeaways")
    return m, d


# Setup --------------------------------------------------------------------------------------------------------
with tabs[0]:

    st.subheader("Objectives")
    st.markdown(
        "- Explore four distinct topic-modelling approaches, ranging from traditional to state-of-the-art, at their default settings.\n"
        "- Evaluate how each model behaves on the same corpus prior to hyperparameter tuning, and identify where each fails out of the box.\n"
        "- Compare all four models under identical conditions using a fixed random seed, the same 10% sample of the cleaned corpus, and uniform evaluation metrics."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Posts modelled", format_int(setup["modelled_posts"]), 
            f"{setup['modelled_fraction']*100:.0f}% random sample", delta_color="off", delta_arrow="off")
    c2.metric("Publications modelled", format_int(setup["modelled_publications"]))
    c3.metric("Models compared", format_int(setup["models_compared"]))
    

    st.subheader("Model overview")
    st.dataframe(
        pd.DataFrame({
            "Model": [MODEL_NAMES[m] for m in MODEL_ORDER],
            "Approach": [
                "Word counts, matrix factorisation",
                "Word counts, probabilistic",
                "Embeddings, clustering",
                "Embeddings, keywords, matrix factorisation",
            ],
            "Input": [
                "Raw text",
                "spaCy lemmas (nouns, proper nouns, adjectives)",
                "Raw text",
                "Raw text",
            ],
        }),
        hide_index=True,
    )

    st.subheader("Evaluation setup and controls")
    st.dataframe(
        pd.DataFrame({
            "Parameter": [
                "Corpus sample",
                "Random seed",
                "Number of topics",
                "Scored words",
                "Co-occurrence reference",
                "Embedding metrics",
                "Word vectors",
            ],
            "Configuration": [
                f"{setup['modelled_fraction'] * 100:.0f}% random sample of the cleaned corpus ({format_int(N_DOCS)} posts), using a fixed seed",
                f"Fixed seed (`{setup['seed']}`)",
                f"{setup['target_num_topics']} topics per model (BERTopic configured to 21 to account for its outlier group)",
                f"{setup['top_n_words_extracted']} words generated per topic, with the first 10 valid words scored",
                f"{setup['n_coherence_reference_docs']:,} posts, tokenised identically across all models for coherence evaluation",
                f"Computed using an independent encoder (`{setup['reference_embeddings_model']}`), different from the encoder used to fit BERTopic and KeyNMF",
                f"Pretrained {setup['word_vectors']}, independent of all models for coherence evaluation",
            ],
        }),
        hide_index=True,
    )

    st.subheader("Evaluation metrics")
    st.dataframe(
        pd.DataFrame([
            {
                "Metric": v.label,
                "Question": v.question,
                "Theoretical range": v.theoretical_range,
                "Typical range": v.typical_range,
                "Interpretation": v.interpretation,
            }
            for v in KEY_METRICS.values()
        ]),
        hide_index=True,
        row_height=80,
    )


# TF-IDF + NMF ---------------------------------------------------------------------------------------------
with tabs[1]:
    m, d = model_tab("nmf")
    rest = d["absolute_share_%"].iloc[1:]

    callouts(
        "observation",
        f"Co-occurrence coherence ({m['coherence_c_npmi']:.3f}) shows that top words appear together more often than random chance.",
        f"However, **Topic 0 holds roughly {m['largest_topic_absolute_%']:.0f}% of all posts** and consists of generic words: "
        f"*people, ve, work, way, make, years*. Each remaining topic accounts for between {rest.min():.1f}% and {rest.max():.1f}%.",
        "Several topics reflect **publishing format and calendar schedules** rather than subject matter. One groups months and "
        "years (*2026, september, august, july, newsletter*), while another centres on weekly cadence (*week, weekly, news, friday, newsletter*).",
        f"Topic diversity is {m['topic_diversity']:.2f}, with common Substack words such as *newsletter* appearing across multiple topics.",
    )


    callouts(
        "takeaway",
        "The token *ve* comes from contractions like *I've*, because the default tokeniser splits words at apostrophes. "
        "With minimal preprocessing, word fragments like this and different forms of the same root word end up cluttering the topic keywords.",
        "Reviewing the topic cards confirms that NMF focuses heavily on exact keywords rather than broader concepts, grouping "
        "format-based topics around shared words like \"book\", \"letter\", \"podcast\", and \"video\".",
        "Stronger text cleaning (such as expanding contractions, lemmatising words, and removing common platform stopwords) "
        "is needed before NMF can uncover clear and meaningful thematic topics.",
    )



# LDA --------------------------------------------------------------------------------------------------------
with tabs[2]:
    m, d = model_tab("lda")

    callouts(
        "observation",
        f"Topic sizes are fairly balanced (ranging from {d['absolute_share_%'].min():.1f}% to {d['absolute_share_%'].max():.0f}% of "
        f"posts), and only {m['outlier_rate']:.1f}% of posts are unassigned (posts left empty after filtering).",
        "Topics are less format-driven and group related concepts together. For example, one topic pairs *AI*, *business*, "
        "and *future*, while another covers *Trump*, *political*, *government*, and *law*, though a calendar topic of month names "
        "still appears.",
        f"Compared with NMF, LDA achieves lower co-occurrence coherence "
        f"({m['coherence_c_npmi']:.3f}) but much higher topic diversity ({m['topic_diversity']:.3f}).",
    )

    callouts(
        "takeaway",
        "LDA surfaces more meaningful, subject-focused themes rather than publishing formats, improving substantially on NMF.",
        "This makes it a practical baseline for exploring high-level themes across the dataset.",
    )

# BERTopic ------------------------------------------------------------------------------------------------------
with tabs[3]:
    m, d = model_tab("bertopic")
    small = int((d["relative_share_%"] < 1).sum())

    callouts(
        "observation",
        f"**{m['outlier_rate']:.0f}% of posts are left unassigned.** Of the {len(d)} resulting topics, {small} each "
        f"account for under 1% of all **assigned** posts.",
        "When BERTopic reduces topics to the target count, the **outlier group** (`-1`) is counted toward that total.",
        f"All evaluation metrics are computed solely on the {100 - m['outlier_rate']:.0f}% of posts assigned to a topic, "
        f"which represent the easiest documents to cluster.",
        f"BERTopic achieves higher document tightness above chance (`centroid_sim_diff` of {m['centroid_sim_diff']:.3f}), "
        f"indicating tighter clusters, though its word-embedding coherence is lower ({m['wec']:.3f}).",
        "Topics are noticeably more concrete, covering specific subjects such as war and politics (*iran, war, israel, trump, "
        "ukraine*), food (*recipe, food, wine, chicken*), and finance (*market, crypto, bitcoin, money*), alongside format "
        "topics (*poem, episode, podcast, music*) and broader life themes (*art, life, love, grief*).",
    )
    callouts(
        "takeaway",
        "Topics capture genuine semantic meaning and offer a balanced mix of broad and niche subjects.",
        "Compared with NMF and LDA, the top words for each topic are far more specific rather than generic everyday terms.",
        "Sample posts within a cluster consistently reflect the topic's theme even without sharing its exact keywords. "
        "For instance, the sports topic has general top words (*game, football, cup, season, league, nfl, baseball, nba*), "
        "yet its assigned posts discuss specific teams and players without using those exact words.",
        "This semantic grouping is especially effective for domains with rich proper nouns and varied terminology (such as "
        "sports teams, players, and coaches), whereas NMF and LDA often group unrelated posts simply because they share an "
        "ambiguous keyword.",
    )

# KeyNMF --------------------------------------------------------------------------------------------------------
with tabs[4]:
    m, d = model_tab("keynmf")

    callouts(
        "observation",
        f"KeyNMF scores highest on word-embedding coherence ({m['wec']:.3f}) and ties for the highest co-occurrence "
        f"coherence (`c_npmi` of {m['coherence_c_npmi']:.3f}).",
        f"However, **Topic 0 holds roughly {m['largest_topic_absolute_%']:.0f}% of all posts**, acting as a catch-all of broad "
        f"words (*world, war, self, future, power, women, trump, time*).",
        f"Topic diversity is the lowest of the four models ({m['topic_diversity']:.3f}), with common words such as *time*, "
        f"*happy*, and *new* repeating across several topics.",
        "Similar to NMF, several topics reflect publishing format, calendar schedules, and occasions (such as years and "
        "birthdays, or *day, mother, valentine, morning*) rather than actual subject matter.",
        f"Its document tightness above chance (`centroid_sim_diff` of {m['centroid_sim_diff']:.3f}) is lower than BERTopic's "
        f"({metrics.loc['bertopic', 'centroid_sim_diff']:.3f}), meaning the clusters are looser, though it still beats "
        f"both NMF and LDA.",
    )
    callouts(
        "takeaway",
        "Like standard NMF, the topic keywords remain high-level and generic rather than specific.",
        "Unlike standard NMF, however, KeyNMF does not require a post to contain the exact topic keywords to be grouped there. "
        "For example, one topic has generic keywords (*things, home, little, month, list, time, joy, learned, feel, 10*), "
        "yet it still groups a post reflecting on spiritual highs and lows after Easter.",
        "As a result, even when the topic keywords look plain, the posts grouped under them fit together well.",
    )


# Side by side -----------------------------------------------------------------------------------------------
with tabs[5]:
    st.caption(
        "Comparing all four models on the same dataset and evaluation metrics. "
        "These results come from a single run, so focus on clear gaps between models and treat tiny differences as ties."
    )

    st.subheader("Quantitative comparison")
    comparison_df = (
        pd.DataFrame({
            meta.short_name: metrics[k] * meta.scale
            for k, meta in KEY_METRICS.items()
        })
        .rename(index=MODEL_NAMES)
        .style.apply(highlight_extremes, gap_ratio=0.15, axis=0)
    )

    st.dataframe(
        comparison_df,
        column_config={
            meta.short_name: st.column_config.NumberColumn(
                format=meta.fmt,
                help=meta.help_text,
            )
            for meta in KEY_METRICS.values()
        },
    )
    callouts(
        "observation",
        "**No single model leads across all metrics.** Each model's strongest score comes with a clear trade-off.",
        "**The highest coherence scores coincide with a catch-all topic.** NMF and KeyNMF score highest on both word-level "
        "metrics, yet both rely on a single topic holding nearly half of all posts. Generic words frequently co-occur "
        "with many other terms, which can inflate coherence scores without producing semantically meaningful or useful topics.",
        "**BERTopic achieves the highest document tightness**, supported by semantic clustering, though this advantage is largely "
        f"driven by leaving {metrics.loc['bertopic', 'outlier_rate']:.0f}% of posts unassigned.",
        "**Topic size distributions vary sharply across models.** LDA is nearly uniform, NMF and KeyNMF "
        f"peak near {metrics.loc['nmf', 'largest_topic_relative_%']:.0f}% and {metrics.loc['keynmf', 'largest_topic_relative_%']:.0f}% " 
        "and BERTopic exhibits a long tail of small topics.",
    )

    st.subheader("Explore")
    t_metrics, t_catch, t_sizes, t_top, t_rank = st.tabs(
        ["Metrics", "Catch-all topic", "Topic sizes", "Largest topics", "Ranking"])
    with t_metrics:
        default = [k for k in KEY_METRICS if k != "runtime_m"]
        chosen = st.multiselect(
            "Select metrics to compare in the visualisations below",
            list(KEY_METRICS),
            default=default,
            format_func=lambda k: KEY_METRICS[k].label,
        )
        if chosen:
            scaled = metrics[chosen].copy()
            for k in chosen:
                scaled[k] = scaled[k] * KEY_METRICS[k].scale

            long = scaled.reset_index().melt("model", var_name="key", value_name="value")
            long["metric"] = long["key"].map(lambda k: KEY_METRICS[k].label)
            long["model"] = long["model"].map(MODEL_NAMES)
          

            n_cols = min(len(chosen), 3)
            n_rows = (len(chosen) + n_cols - 1) // n_cols

            # Scale height per row generously and adjust spacing dynamically
            row_height = 280 if n_rows == 1 else 250
            total_height = row_height * n_rows + 80
            v_spacing = 0.10 if n_rows > 1 else 0.1

            fig = px.bar(
                long,
                x="value",
                y="model",
                color="model",
                facet_col="metric",
                facet_col_wrap=n_cols,
                facet_row_spacing=v_spacing,
                facet_col_spacing=0.08,
                orientation="h",
                template=PLOT_TEMPLATE,
                color_discrete_map=COLOURS,
                category_orders={
                    "model": [MODEL_NAMES[k] for k in MODEL_ORDER]
                },
                text_auto=".2g",
            )
            fig.update_xaxes(matches=None, showticklabels=True, title=None)
            fig.update_yaxes(title=None)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_layout(
                height=total_height, showlegend=False, margin=dict(t=50, b=30, r=30)
            )
            st.plotly_chart(fig, width="stretch")
    with t_catch:
        sc = metrics.reset_index()
        sc["name"] = sc["model"].map(MODEL_NAMES)

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.caption("**Co-occurrence Coherence vs. Largest Topic**")
                fig_npmi = px.scatter(
                    sc,
                    x="largest_topic_relative_%",
                    y="coherence_c_npmi",
                    color="name",
                    text="name",
                    template=PLOT_TEMPLATE,
                    color_discrete_map=COLOURS,
                    hover_data={"outlier_rate": ":.2f", 
                                "name": True, 
                                "largest_topic_relative_%": ":.1f", 
                                "coherence_c_npmi": ":.3f", 
                                },
                    labels={
                        "largest_topic_relative_%": "Largest topic, % of assigned posts",
                        "coherence_c_npmi": "Co-occurrence coherence (c_npmi)",
                        "outlier_rate": "Outlier rate (%)",
                    },
                )
                fig_npmi.update_traces(
                    marker=dict(size=16), textposition="top center", cliponaxis=False
                )
                fig_npmi.update_layout(height=380, showlegend=False, margin=dict(t=25, b=10, l=10, r=15))
                st.plotly_chart(fig_npmi, width="stretch")

        with col2:
            with st.container(border=True):
                st.caption("**Embedding Coherence vs. Largest Topic**")
                fig_wec = px.scatter(
                    sc,
                    x="largest_topic_relative_%",
                    y="wec",
                    color="name",
                    text="name",
                    template=PLOT_TEMPLATE,
                    color_discrete_map=COLOURS,
                    hover_data={"outlier_rate": ":.2f", 
                                "name": True, 
                                "largest_topic_relative_%": ":.1f", 
                                "wec": ":.3f", 
                                },
                    labels={
                        "largest_topic_relative_%": "Largest topic, % of assigned posts",
                        "wec": "Word-embedding coherence (WEC)",
                        "outlier_rate": "Outlier rate (%)",
                    },
                )
                fig_wec.update_traces(
                    marker=dict(size=16), textposition="top center", cliponaxis=False
                )
                fig_wec.update_layout(height=380, showlegend=False, margin=dict(t=25, b=10, l=10, r=15))
                st.plotly_chart(fig_wec, width="stretch")

        st.caption(
            "The two models achieving the highest coherence also rely on a single topic holding nearly half of all posts."
        )

    with t_sizes:
        curve = cards.copy()
        curve["rank"] = curve.groupby("model")["size"].rank(ascending=False, method="first").astype(int)
        curve["name"] = curve["model"].map(MODEL_NAMES)
        st.markdown("###### Topic size distribution across models")

        fig = px.line(
            curve.sort_values("rank"), 
            x="rank", 
            y="relative_share_%", 
            color="name", 
            markers=True,
            template=PLOT_TEMPLATE, 
            color_discrete_map=COLOURS, 
            hover_data={"top_words": True},
            labels={"name": "Model",
                    "rank": "Topic rank by size", 
                    "relative_share_%": "% of assigned posts", 
                    })
        fig.update_xaxes(dtick=1)
        fig.update_layout(height=380, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        st.caption(
                    "**TF-IDF + NMF and KeyNMF follow nearly identical distributions.** "
                    "Hover over a point for additional details. "
                    "Double-click a legend item to isolate a model, or double-click again to reset."
                )
        
    with t_top:
        n = st.slider("Number of largest topics to display per model", 3, 20, 5, key="n_side")
        cols = {"Rank": range(1, n + 1)}
        for model in MODEL_ORDER:
            top = (
                cards[cards["model"] == model]
                .sort_values("size", ascending=False)
                .head(n)
            )
            words = top["top_words"].str.split(", ").str[:10].str.join(", ")
            cols[MODEL_NAMES[model]] = [
                f"**{s:.1f}%** | {w}" for s, w in zip(top["relative_share_%"], words)
            ]

        st.table(pd.DataFrame(cols), hide_index=True)
        st.caption(
            "Relative share of assigned posts per topic, followed by each topic's top 10 words."
        )

    with t_rank:
        RANK_METRICS = [k for k in KEY_METRICS if k not in ("largest_topic_absolute_%", "runtime_m")]

        ranks = {}
        for k in RANK_METRICS:
            meta = KEY_METRICS[k]
            ranks[meta.label] = (
                metrics[k]
                .rank(ascending=meta.lower_is_better, method="min")
                .astype(int)
            )

        rank_df = pd.DataFrame(ranks).T[MODEL_ORDER].rename(columns=MODEL_NAMES)

        # Add an overall summary row at the bottom
        rank_df.loc["Average rank"] = rank_df.mean().round(2)

        def style_ranks(row: pd.Series) -> list[str]:
            if row.name == "Average rank":
                best = row.min()
                return [
                    "font-weight: bold; background-color: rgba(33, 195, 84, 0.15);"
                    if v == best
                    else "font-weight: bold;"
                    for v in row
                ]
            return [
                "color: #21c354; font-weight: bold;"
                if v == 1
                else ("color: #ff4b4b;" if v == len(MODEL_ORDER) else "")
                for v in row
            ]

        st.markdown("###### Model rankings across key metrics")
        st.dataframe(
            rank_df.style.apply(style_ranks, axis=1).format(
                lambda v: f"{v:.2f}"
                if isinstance(v, float) and not v.is_integer()
                else f"{int(v)}"
            ),
        )
        st.caption(
            "1 = best (excludes `largest_topic_absolute_%` and runtime). Average rank is a fairly crude measure, "
            "as it is an unweighted mean across the included metrics. "
            "Outlier rate is ranked lower-is-better, which rewards forcing every post into a topic, so interpret it alongside "
            "largest topic share and coherence."
        )


    with st.expander("Iterations: what changed along the way"):
        st.markdown(
            "- **Standardised word counts for coherence.** Initial coherence scores were not directly comparable because the "
            "scoring function truncated all topics to match the shortest one. Models were evaluated on anywhere from 2 to "
            "8 words per topic, which artificially boosted NMF since it was scored on only 4 words. Every topic is now "
            "evaluated on a fixed set of 10 words, with explicit reporting of how many words and topics are scored.\n"
            "- **Dropped redundant metrics.** `coherence_c_v` ranks the models in a nearly identical order to `c_npmi` (aside from "
            f"a 0.001 tie at the top), while silhouette scores fall within a narrow band between {metrics['silhouette'].min():.2f} and "
            f"{metrics['silhouette'].max():.2f} across all models. Neither adds a distinct signal, so both were excluded.\n"
            "- **Excluded ground-truth labels.** No evaluation metric relies on ground-truth labels, as the `category` column is generated by "
            "broad regex rules (see the Data exploration page)."
        )


# Takeaways --------------------------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Failure modes at default settings and next steps")
    small = int((cards[cards["model"] == "bertopic"]["relative_share_%"] < 1).sum())
    st.dataframe(
        pd.DataFrame([
            {
                "Problem": "Dominant catch-all topic (NMF, KeyNMF)",
                "Evidence": (
                    f"Roughly {metrics.loc['nmf', 'largest_topic_absolute_%']:.0f}% and "
                    f"{metrics.loc['keynmf', 'largest_topic_absolute_%']:.0f}% of all posts fall into Topic 0, "
                    "dominated by generic words"
                ),
                "Next step (Notebook 02)": "Sweep the number of topics, and re-score coherence with the catch-all topic excluded.",
            },
            {
                "Problem": "High outlier rate (BERTopic)",
                "Evidence": (
                    f"{100 * metrics.loc['bertopic', 'outlier_rate']:.1f}% of posts left unassigned, and {small} of 20 topics "
                    "each account for under 1% of assigned posts"
                ),
                "Next step (Notebook 02)": "Sweep `min_cluster_size`, then apply outlier reduction and compare models on the same post subset.",
            },
            {
                "Problem": "Tokenisation artefacts",
                "Evidence": "Contraction fragments such as *ve* and *don* appear among top topic words",
                "Next step (Notebook 02)": "Adjust the token pattern to preserve contractions or expand the stopword list.",
            },
            {
                "Problem": "Single random seed (no variance estimate)",
                "Evidence": "Narrow score gaps across several evaluation metrics",
                "Next step (Notebook 02)": "Run across 3 to 5 random seeds before establishing any model ranking.",
            },
            {
                "Problem": "Calendar and format topics across all models",
                "Evidence": "Month names, years, \"weekly\", and \"newsletter\" dominate several topics, reflecting 82% of posts coming from 2025 to 2026",
                "Next step (Notebook 02)": "Decide whether to strip month and year tokens or retain them as a substantive corpus finding.",
            },
        ]),
        hide_index=True,
    )
    callouts(
        "takeaway",
        "On automated metrics alone, NMF and KeyNMF look like the top choices, but reading the actual topics rules both out.",
        "For NMF and KeyNMF, nearly half of all posts sit inside a single catch-all topic, while most remaining topics reflect generic words, publishing formats, and calendar dates rather than actual subjects.",
        "LDA and BERTopic find much more meaningful themes, though each has a clear trade-off: LDA has the weakest word co-occurrence, while BERTopic leaves roughly half of all posts unassigned.",
        "This gap between automated scores and manual reading is common in real-world topic modelling projects, and directly guides the tuning strategy in Notebook 02.",
    )

