"""Home: Project overview, headline figures, and a guide to navigating the notebooks."""

import streamlit as st

from utils import MODEL_NAMES, load_csv, load_json, get_artefact, format_int

st.set_page_config(page_title="Topic modelling on Substack posts", page_icon="🧭", layout="wide")

overview = get_artefact(load_json, "00_data_exploration", "corpus_overview.json")
model_setup = get_artefact(load_json, "01_models_default_settings", "model_setup.json")
funnel = get_artefact(load_csv, "00_data_exploration", "corpus_funnel.csv")
metrics = get_artefact(load_csv, "01_models_default_settings", "model_default_metrics.csv").set_index("model")

st.title("What are 3.5 million Substack posts about?")
st.markdown(
    """
    **This project evaluates four topic-modelling approaches on Substack newsletter titles and subtitles 
    to establish a reliable thematic map of what writers publish and uncover broader content patterns.**
    Adapted from exploratory Jupyter notebooks, this interactive application walks through how methodological decisions 
    were made rather than merely presenting the final results. It highlights key data observations, underlying assumptions, takeaways, and instances 
    where quantitative metrics diverged from qualitative review.
    """
)

st.subheader("Progress to Date")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts in the raw dataset", format_int(overview["raw_posts"]))
c2.metric("After cleaning", format_int(overview["cleaned_posts"]),
          f"{overview['pct_cleaned_posts']:.0f}% kept", delta_color="off", delta_arrow="off")
c3.metric("Posts modelled", format_int(model_setup["modelled_posts"]), 
          f"{model_setup['modelled_fraction']*100:.0f}% random sample", delta_color="off", delta_arrow="off")
c4.metric("Models compared", format_int(model_setup["models_compared"]))

st.subheader("Key Insights from Initial Analysis")
nmf, key = metrics.loc["nmf"], metrics.loc["keynmf"]
language_cut = funnel.set_index("step").loc["English & Latin script", "dropped_pct_of_previous"]
bert = metrics.loc["bertopic"]
st.markdown(
    f"""
1. **Data cleaning removed approximately {100 - overview['pct_cleaned_posts']:.0f}% of the dataset**, 
   driven primarily by language filtering ({language_cut:.0f}%). 
   While {100 - language_cut:.0f}% of the posts are detected as English, short titles present challenges for accurate language detection.
2. **Initial modelling at default settings reveals a trade-off between topic coherence and topic coverage:**
   - **High Coherence, Broad Catch-Alls:** {MODEL_NAMES['nmf']} and {MODEL_NAMES['keynmf']} (Turftopic) achieve the highest word coherence scores, 
     yet both rely on a single generic topic that absorbs {nmf['largest_topic_absolute_%']:.0f}% and {key['largest_topic_absolute_%']:.0f}% of all posts, respectively.
   - **Concrete Topics, High Noise:** {MODEL_NAMES['bertopic']} yields more distinct, interpretable topics upon manual review, 
     but leaves {100 * bert['outlier_rate']:.0f}% of the posts unassigned as outliers.
3. **As no single model currently outperforms across all evaluation metrics, selecting a final approach is premature.** 
   The next phase will focus on hyperparameter tuning guided by a clearly defined optimisation objective.
"""
)


st.subheader("How to Navigate This App")
st.markdown(
    """
    Each page corresponds to an individual notebook. Within each page, every step is structured around three core notes:
- **Observations**: The numbers and patterns observed in the data.
- **Decisions**: The actions taken based on the observations, and the reasoning behind them.
- **Key Takeaways**: The main insights and lessons learned from the step.
"""
)

st.subheader("Roadmap")
st.dataframe(
    {
        "Step": ["Data exploration", 
                 "Models at default settings", 
                 "Models Tuning", 
                 "Model comparison (5 seeds)",
                 "Topic review (hierarchy, DataMapPlot)", 
                 "Seeded topic modelling"],
        "Notebook": ["00", "01", "02", "03", "04", "05"],
        "Status": ["Done", "Done", "Next", "Planned", "Planned", "Planned"],
    },
    hide_index=True, 
)
st.caption(
    """**Data Source:** Substack Newsletter Articles & Metadata (3.5M+)
    (uploaded by reja37 on Kaggle: https://www.kaggle.com/datasets/reja37/substack-newsletter-articles-and-metadata-3-5m/data) 
    \n**Citation**: Karim, Rejaul (2026), *“Substack Articles & Newsletter Metadata Dataset”*, 
    Mendeley Data, V1, doi: 10.17632/btvzfpd33f.1
    """
)
