import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
import pandas as pd
import math
CUSTOM_HEX = ["#83241E", "#102020", "#78B4B7","#2D4C4F", "#D9E9E9"]

from substack_topics.metrics import get_topic_counts

def apply_custom_theme(figsize=(20, 10), nrows=1, ncols=1, palette=CUSTOM_HEX):
    """Initializes the multi-plot canvas window grid cleanly once."""
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    sns.set_theme(style="whitegrid", palette=palette, rc={"axes.grid": False})
    # Flatten axes array for simple 1D index looping if it's a matrix
    return fig, axes.flatten() if hasattr(axes, 'flatten') else (fig, [axes])

def finalize_plot_layout(ax, title: str, xlabel: str, ylabel: str):
    """Handles text sizes, labels, grid lines, and rendering wrappers uniformly."""

    ax.set_title(title, fontsize=14, pad=15, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

def plot_line_chart_subplot(
    ax,  # Pass the specific subplot axis from your grid loop
    df,
    x,
    y,  # Can be a string or a list of strings
    hue=None,
    palette=CUSTOM_HEX,
    title="",
    ylabel="",
    xlabel="",
    vline_value=None,
    vline_label="---",
):
    """Plots a standard single-axis line chart inside a specific subplot axis slot."""
    
    # 1. Match vertical line type if present on the targeted axis
    if vline_value is not None:
        ax.axvline(vline_value, color='grey', linestyle='--', label=vline_label)

    # Ensure y is always a list to simplify looping
    y_list = [y] if isinstance(y, str) else list(y)

    # 2. Render plot lines directly on the provided axis layer (ax=ax)
    if hue is None:
        for i, y_col in enumerate(y_list):
            color = (
                palette[i % len(palette)]
                if isinstance(palette, list)
                else palette
            )
            sns.lineplot(
                data=df,
                x=x,
                y=y_col,
                marker="o",
                color=color,
                ax=ax,          # Bound to current subplot
                label=y_col,    # Labels the line with its column name
            )
    else:
        for y_col in y_list:
            sns.lineplot(
                data=df, x=x, y=y_col, marker="o", hue=hue, palette=palette, ax=ax
            )

    # 3. Handle standard text formatting and layout rules uniformly
    finalize_plot_layout(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_dual_axis_subplot(
    ax1,  
    df,
    x,
    y1,
    y2,
    palette=CUSTOM_HEX,
    title="",
    ylabel1="",
    ylabel2="",
    xlabel="",
    vline_value=None,
    vline_label="---",
):
    """Plots a dual-axis line chart inside a single subplot axis of a larger grid."""
    
    # 1. Match vertical line type if present
    if vline_value is not None:
        ax1.axvline(
            vline_value,
            color="grey",
            linestyle="--",
            label=vline_label,
        )

    # 2. Plot lines on their respective axes layers
    sns.lineplot(
        data=df, x=x, y=y1, marker="o", color=palette[0], ax=ax1, label=y1
    )
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df, x=x, y=y2, marker="o", color=palette[1], ax=ax2, label=y2
    )

    # 3. Finalize general layout using the helper function on ax1
    finalize_plot_layout(ax1, title=title, xlabel=xlabel, ylabel=ylabel1)

    # 4. Apply custom dual-axis color branding overrides
    ax1.yaxis.label.set_color(palette[0])
    ax1.yaxis.label.set_weight("semibold")
    ax1.tick_params(axis="y", labelcolor=palette[0])

    ax2.set_ylabel(ylabel2, fontsize=12, color=palette[1], fontweight="semibold")
    ax2.tick_params(axis="y", labelcolor=palette[1])

    # 5. Consolidate legends from both layers into a single box
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    if ax1.get_legend(): ax1.get_legend().remove()
    if ax2.get_legend(): ax2.get_legend().remove()
    
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")



def plot_horizontal_bar_chart_subplot(
    ax,  # Pass the specific subplot axis from your grid loop
    df,
    y,  # The category column on the vertical axis (string)
    x,  # The metric value(s) on the horizontal axis (string or list of strings)
    hue=None,
    palette=CUSTOM_HEX,
    title="",
    ylabel="",
    xlabel="",
    vline_value=None,
    vline_label="---",
):
    """Plots a standard horizontal bar chart inside a specific subplot axis slot."""
    
    # 1. Match vertical line type if present on the targeted axis
    if vline_value is not None:
        ax.axvline(
            x=vline_value, color="grey", linestyle="--", label=vline_label
        )

    # Ensure x is always treated as a list for multiple metrics
    x_list = [x] if isinstance(x, str) else list(x)

    # 2. Render bar plots directly on the provided axis layer (ax=ax)
    if hue is None:
        if len(x_list) == 1:
            # Single metric bar chart
            color = palette[0] if isinstance(palette, list) else palette
            sns.barplot(
                data=df, y=y, x=x_list[0], color=color, ax=ax, label=x_list[0]
            )
        else:
            # Multi-metric: Melt data to create grouped side-by-side horizontal bars
            melted_df = df.melt(
                id_vars=[y],
                value_vars=x_list,
                var_name="Metric",
                value_name="Value",
            )
            sns.barplot(
                data=melted_df,
                y=y,
                x="Value",
                hue="Metric",
                palette=palette[:len(x_list)] if isinstance(palette, list) else palette,
                ax=ax,
            )
    else:
        # Standard grouping by a secondary dataframe categorical column
        sns.barplot(data=df, y=y, x=x_list[0], hue=hue, palette=palette, ax=ax)

    # 3. Handle standard text formatting and layout rules uniformly
    finalize_plot_layout(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_facet_line_chart(
    df,
    x,
    y,
    col,
    palette=CUSTOM_HEX,  # Matches the format of your line chart
    title="",
    ylabel="",
    xlabel="",
    col_wrap=4,
    height=4,
    sharey=False,
    vline_value=None,
    vline_label="---",
):
    # Set theme before creating the figure canvas
    sns.set_theme(style="whitegrid", palette=palette, rc={"axes.grid": False})

    # Create the facet grid layout
    g = sns.FacetGrid(
        df, col=col, col_wrap=col_wrap, height=height, sharey=sharey
    )

    color = palette[0] if isinstance(palette, list) else palette
    
    # Map the lineplot explicitly passing data structure columns
    g.map_dataframe(
        sns.lineplot, x=x, y=y, marker="o", color=color
    )

    # Draw vertical milestones directly on the axes to avoid map bloating
    if vline_value is not None:
        for ax in g.axes.flat:
            ax.axvline(
                vline_value,
                color="grey",
                linestyle="--",
                linewidth=1.5,
                label=vline_label,
            )

    # Adjust labels and titles cleanly
    g.set_axis_labels(xlabel, ylabel)
    if title:
        g.fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Topic Modeling (Individual Models) ---------------------------------------------------------------------------
def plot_topic_distributions(doc_topics: np.ndarray, top_n_topics: int | None = None, model_name: str = "") -> None:
    """Plots a beautiful horizontal Seaborn bar chart tracking topic sizes and shares.

    Args:
        doc_topics: An array containing the assigned topic ID for each document.
        top_n_topics: The maximum number of top topics to display (sorted by size). 
                      Defaults to None, which displays all topics.
        model_name: The name of the model to display as the title. Defaults to an empty string.
                      
    """
    # 1. Standardise and dynamically extract frequencies
    d = np.asarray(doc_topics)
    sizes, assigned = get_topic_counts(d, top_n_topics if top_n_topics is not None else 0)
    percentages = (sizes / assigned) * 100

    # 2. Build DataFrame and sort by size
    df = pd.DataFrame(
        {
            "Topic ID": [f"Topic {i}" for i in range(len(sizes))],
            "Document Count": sizes,
            "Share (%)": percentages,
        }
    )
    
    # Sort from largest to smallest
    df = df.sort_values(by="Document Count", ascending=False)
    
    # Slice only if a specific restriction is passed down
    if top_n_topics is not None:
        df = df.head(top_n_topics)

    # 3. Apply Seaborn styling aesthetics
    sns.set_theme(style="whitegrid", font="monospace")
    
    
    # Dynamic sizing prevents massive bars when showing all topics
    plot_height = max(4, len(df) * 0.5)
    fig, ax1 = plt.subplots(figsize=(10, plot_height))

    # Primary X-Axis (Bottom): Absolute Document Counts
    sns.barplot(
        x="Document Count",
        y="Topic ID",
        data=df,
        ax=ax1,
        color="#83241E",
        edgecolor=".2",
        orient="h",
    )

    ax1.set_ylabel("")
    ax1.set_xlabel("")

    # Secondary X-Axis (Top): Share Percentage
    ax2 = ax1.twiny()
    ax2.set_xlabel("")

    # Mirror the percentage grid perfectly using the filtered max value
    max_count = df["Document Count"].max() if len(df) > 0 else 1
    ax2.set_xlim(0, (max_count / assigned) * 110)
    ax2.grid(False)

    # Clear numbers/labels
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    
    # Hide the tick mark dashes
    ax1.tick_params(bottom=False)
    ax2.tick_params(top=False)
    
    # Remove spines
    sns.despine(ax=ax1, bottom=True, left=False)
    sns.despine(ax=ax2, top=True)

    # 4. Add text annotations cleanly matching the new sorted indices
    for index, (_, row) in enumerate(df.iterrows()):
        ax1.text(
            x=row["Document Count"],
            y=index,
            s=f" {int(row['Document Count']):,} ({row['Share (%)']:.1f}%)",
            color="black",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="normal",
        )

    # Determine dynamic title based on whether we filtered or showed all
    title_scope = f"Top {len(df)}" if top_n_topics is not None else "All"
    if model_name:
        title = f"{model_name.upper()}: {title_scope}"
    else:
        title = title_scope

    fig.suptitle(
        f"{title} Topic Size Distributions (Total Assigned: {assigned:,})",
        fontsize=16,
        fontweight="bold",
        x=0.5,
        y=1.02,
        ha="center",
    )

    fig.text(
        x=0.5,
        y=0.97,  
        s="Percentage represents share of assigned documents",
        fontsize=11,
        fontstyle="italic",
        color="dimgray",  
        ha="center",
    )
    fig.tight_layout()
    plt.show()

def inspect_model_topics(
    result, 
    texts: list[str], 
    n_docs: int = 3, 
    topn: int = 10, 
    seed: int = 0, 
    max_chars: int = 200
) -> pd.DataFrame:
    
    """One row per topic: size, top words and a few example documents. Read these before trusting any metric.

    Args:
        result: The fitted topic model result containing topics and document-topic assignments.
        texts: The original list of raw text string.
        n_docs: Number of example documents to show per topic.
        topn: Number of top words to display per topic.
        seed: Random seed for selecting example documents.
        max_chars: Maximum number of characters to display for each example document.
    """

    d = np.asarray(result.doc_topics)
    sizes, assigned = get_topic_counts(d, len(result.topics))
    total_len = max(len(d), 1)
    
    rng = np.random.default_rng(seed)
    rows = []
    
    for t, words in enumerate(result.topics):
        # Get the indices of documents assigned to this topic
        idx = np.flatnonzero(d == t)
        pick = rng.choice(idx, size=min(n_docs, len(idx)), replace=False) if len(idx) else []
        
        rows.append({
            "topic": t, 
            "size": int(sizes[t]), 
            "relative_share_%": round(100 * sizes[t] / assigned, 1),
            "absolute_share_%": round(100 * sizes[t] / total_len, 1),
            "top_words": ", ".join(words[:topn]),
            "examples": " || ".join(texts[i][:max_chars] for i in pick),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)

# Topic Modeling (Models Comparison) ---------------------------------------------------------------------------
METRIC_LABELS = {
    "coherence_c_v": "Coherence c_v\n(higher = better)",
    "coherence_c_npmi": "Coherence c_npmi\n(higher = better)",
    "wec": "Word Embedding Coherence\n(higher = better)",
    "topic_diversity": "Topic diversity\n(higher = better)",
    "outlier_rate": "Outlier rate\n(share of docs unassigned)",
    "runtime_m": "Runtime, minutes",
    "centroid_sim": "Centroid similarity\n(higher = better)",
    "centroid_sim_diff": "Centroid similarity difference from chance\n(higher = better)",
    "silhouette": "Silhouette score\n(higher = better)",
}

def plot_model_comparison(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = ("topic_diversity", "outlier_rate", "coherence_c_v", "coherence_c_npmi", "runtime_m"),
    log_metrics: tuple[str, ...] = (),
    metric_labels: dict[str, str] = None,
):
    """
    Plots a comparison of different topic models based on various metrics.
    
    Args:
        df: DataFrame containing the metrics for each model, indexed by model name.
        metrics: Tuple of metric names to plot.
        log_metrics: Tuple of metric names to display on a logarithmic scale.
        metric_labels: Dictionary mapping metric names to display labels.
    Returns:
        None. Displays the comparison plot. 
        Panels have their own scales because the metrics have different units
    
    """
    models = list(df.index)
    colours = CUSTOM_HEX
    
    # 1. Calculate how many rows are needed for exactly 4 columns
    n_cols = 4
    n_rows = math.ceil(len(metrics) / n_cols)
    
    # 2. Adjust figsize dynamically based on grid structure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows), squeeze=False)
    axes_flat = axes.flatten()  # Flatten the 2D grid array into a single 1D index line

    for idx, m in enumerate(metrics):
        ax = axes_flat[idx]
        vals = df[m].astype(float).to_numpy()
        
        ax.barh(models, vals, color=[colours[i % len(colours)] for i in range(len(models))])
        # Flips the y-axis so that the first model is on top
        ax.invert_yaxis()                                  
        if m in log_metrics:
            ax.set_xscale("log")

        if metric_labels is None:
            metric_labels = METRIC_LABELS
        ax.set_title(metric_labels.get(m, m), fontsize=10)
        
        # Printing values on the bars
        for y, v in enumerate(vals):
            if np.isfinite(v):
                ha_align = "left" if v >= 0 else "right"
                # Strip leading space from string if it's negative since 'ha="right"' handles spacing
                text_str = f" {v:.2f}" if abs(v) < 100 else f" {v:.0f}"
                if v < 0:
                    text_str = text_str.strip() + " "
                    
                ax.text(v, y, text_str, va="center", ha=ha_align, fontsize=9)

        # Calculate dynamic limits to add 25% padding to the active side(s)
        xmin, xmax = ax.get_xlim()
        data_range = xmax - xmin
        
        if xmin < 0:
            xmin -= data_range * 0.25                      
        if xmax > 0:
            xmax += data_range * 0.25                      
            
        ax.set_xlim(xmin, xmax)
        ax.spines[["top", "right"]].set_visible(False)
        
        # Only keep y-labels for subplots that sit on the left-most edge (Column 0)
        if idx % n_cols != 0:
            ax.set_yticklabels([])                         

    # Clean up: Hide any empty axes panels if len(metrics) is not a perfect multiple of 4
    for empty_idx in range(len(metrics), len(axes_flat)):
        axes_flat[empty_idx].set_visible(False)

    fig.tight_layout()
    plt.show()


def compare_top_topics(results: dict, n: int = 5, n_words: int = 5) -> pd.DataFrame:
    """Get the top topics across multiple models.
    Args:
        results (dict): Dictionary mapping model names to TopicResult objects.
        n (int, optional): Number of top topics to retrieve for each model. Defaults to 5.
        n_words (int, optional): Number of top words to display for each topic. Defaults to 5.
    
    Returns:
        pd.DataFrame: DataFrame with the top topics for each model, showing the share of assigned documents and top words.
    """
    cols = {}
    for name, r in results.items():
        sizes, assigned = get_topic_counts(r.doc_topics, len(r.topics))
        # Select the top n topics based on their sizes (based on their index in the sorted order)
        order = np.argsort(sizes)[::-1][:n]

        # Calculate the relative percentage share of assigned documents for each top topic
        cols[name] = [
            f"{100 * sizes[t] / assigned:.1f}% | " + ", ".join(r.topics[t][:n_words])
            for t in order
        ]
        # Pad if a model has fewer than n topics
        cols[name] += [""] * (n - len(cols[name]))  

    # Organise dictionary into a vertical matrix where each model name becomes a column and each row represents a top topic rank
    return pd.DataFrame(cols, index=pd.RangeIndex(1, n + 1, name="size rank"))