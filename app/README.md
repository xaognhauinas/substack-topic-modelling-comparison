# Substack topics: Streamlit app

An interactive write-up of a topic-modelling project on 3.5 million Substack post titles and subtitles. The app shows how the analysis unfolded: the data exploration, the cleaning decisions, and four topic models at default settings. It reads small files exported by the notebooks and never fits a model, so it deploys with a short requirements file.

## Contents

| File | Purpose |
|---|---|
| `Home.py` | Project overview, headline figures, key insights, a guide to the notes and the roadmap |
| `pages/1_Data_Exploration.py` | Notebook 00, from the raw dataset to the cleaned corpus |
| `pages/2_Models_Default_Settings.py` | Notebook 01, four topic models compared at default settings |
| `utils.py` | Artefact loaders, the note helpers, model names and colours, and the metric definitions |
| `requirements.txt` | Packages needed to run the app |

The artefacts sit one level up, in `artefacts/`, with one folder per notebook.

## Run the app

Run these commands from the repository root:

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```

## How the app is organised

Each page follows one notebook. Every step combines three kinds of note, shown as badges:

- **Observation**: The numbers and patterns observed in the data.
- **Decision**: The action taken, and the reasoning behind it.
- **Key Takeaway**: The main insight or lesson from the step.


| Page | Tabs |
|---|---|
| Home | *Single-page overview* |
| Data exploration | Overview, Dates, Categories, Publications, Text quality, Duplicates, Language, Low-information posts, Funnel and decisions |
| Models at default settings | Setup, TF-IDF + NMF, LDA, BERTopic, KeyNMF, Side by side, Takeaways |

Figures in the commentary are dynamically populated from `artefacts/` wherever possible, ensuring that re-running the notebooks automatically updates both the narrative text and the visualisations.


## Artefacts

Each notebook ends with an export cell that writes its outputs to a dedicated folder under `artefacts/`. If any expected file is missing, the corresponding Streamlit page stops and names the missing file.

| Folder | Files |
|---|---|
| `00_data_exploration/` | `corpus_overview.json`, `corpus_funnel.csv`, `yearly_posts.csv`, `categories.csv`, `category_by_year.csv`, `text_quality_flags.csv`, `combined_text_word_count.csv`, `combined_text_word_count_summary.csv`, `duplicates_summary.csv`, `language_shares.csv`, `script_shares.csv`, `low_information_summary.csv`, `dataset_dtypes.csv`, `dataset_stats.csv`, `dataset_sample.csv` |
| `01_models_default_settings/` | `model_setup.json`, `model_default_metrics.csv`, `model_default_topic_cards.csv` |


The pages look up JSON keys and CSV columns by name, so renaming a field in a notebook requires updating its reference in the corresponding page script.


## Data source and licence

**Dataset:** Substack Articles & Newsletter Metadata Dataset, by Rejaul Karim, Mendeley Data, Version 1, published 29 September 2026. DOI: [10.17632/btvzfpd33f.1](https://doi.org/10.17632/btvzfpd33f.1). The dataset is licensed under CC BY 4.0, which requires attribution. The citation appears on the Home page.

**What the app displays from the posts:**

- Author names, article URLs, and publication domains are never displayed, as publication addresses often contain the author's handle.
- Sample rows and example posts in the topic tables are truncated for readability. They are sampled at random rather than selected to favour any model.
- The underlying posts were written by Substack authors, any title or snippet can be removed from `artefacts/` upon request.


## Deployment (Streamlit Community Cloud)

Set the main file path to `app/Home.py`. Streamlit Community Cloud installs dependencies from the first requirements file it detects, searching the main file's directory (`app/`) first before checking the repository root. Placing a lightweight `requirements.txt` inside `app/` ensures it takes precedence over the full modelling environment, keeping deployment fast and focused strictly on the packages required by the app.


## Status

Notebooks 00 and 01 are complete. Model tuning (02), the five-seed comparison (03), topic review (04) and seeded topic modelling (05) are planned.