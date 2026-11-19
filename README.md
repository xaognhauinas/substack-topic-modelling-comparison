# Substack Topic Modelling Comparison

## Overview

This project evaluates and compares four topic-modelling approaches on a dataset of **3.5 million Substack posts** to establish a reliable thematic map of newsletter content and uncover broader publishing patterns.

**Goal:** Compare methodologies, evaluate performance metrics, and determine which topic modelling approach best captures meaningful themes in Substack newsletter titles and subtitles.

## Key Features

- **Four Topic Models**: Compare NMF (TF-IDF), LDA (Gensim), BERTopic, and KeyNMF (Turftopic)
- **Transparent Methodology**: Interactive Streamlit application explaining methodological decisions
- **Comprehensive Evaluation**: Multiple evaluation metrics including coherence, diversity, and manual assessment
- **Large-Scale Dataset**: Analysis of 3.5M posts with rigorous data cleaning pipeline
- **Reproducible Analysis**: Jupyter notebooks and modular Python code for complete transparency

## Project Structure

```
substack-topic-modelling-comparison/
├── app/                       # Streamlit dashboard
│   ├── Home.py                # Main dashboard page
│   ├── README.md              # Dashboard documentation
│   ├── pages/                 # Additional dashboard pages
│   ├── utils.py               # App utilities
│   └── requirements.txt       # App dependencies
├── artefacts/                 # Pre-computed results (CSV, JSON)
├── data/
│   ├── raw/                   # Original dataset (not committed)
│   └── interim/               # Processed intermediate data (not committed)
├── notebooks/
│   ├── 00_data_exploration.ipynb           # Dataset overview and cleaning
│   └── 01_models_default_settings.ipynb    # Model training and evaluation at default settings
├── src/substack_topics/
│   ├── __init__.py
│   ├── data/                  # Data utilities and mappings
│   ├── models.py              # Topic model implementations (NMF, LDA, BERTopic, KeyNMF)
│   ├── metrics.py             # Evaluation metrics (coherence, diversity, etc.)
│   ├── embeddings.py          # Embedding generation for semantic models
│   ├── tokenise.py            # Text tokenisation and preprocessing
│   └── plots.py               # Visualization utilities
├── .gitignore                 # Git ignore rules
├── models.yaml                # Default model hyperparameters
├── models_tuned.yaml          # Tuned model hyperparameters and results
├── pyproject.toml             # Project configuration and dependencies
└── README.md                  # Project documentation
```

## Dataset

- **Source:** Kaggle Substack dataset
- **Size:** 3.5 million posts (titles and subtitles)
- **Cleaning:** Language detection, text normalisation, duplicate removal
- **Final corpus:** ~2.8M posts after cleaning (80% retention)

## Topic Models Compared

### 1. TF-IDF + NMF
- **Library:** scikit-learn
- **Approach:** Matrix factorisation of term-frequency-inverse-document-frequency vectors
- **Strengths:** Fast, interpretable, good baseline
- **Configuration:** Scikit-learn default NMF with TF-IDF vectorisation

### 2. LDA (Latent Dirichlet Allocation)
- **Library:** Gensim
- **Approach:** Probabilistic generative model
- **Strengths:** Classic approach, probabilistic topic distributions
- **Configuration:** Gensim LDA with preprocessed, lemmatised tokens generated using spaCy

### 3. BERTopic
- **Library:** BERTopic
- **Approach:** Semantic embeddings + clustering (UMAP + HDBSCAN)
- **Strengths:** Captures semantic meaning, modern deep learning approach
- **Configuration:** Pre-trained sentence transformers with dynamic clustering

### 4. KeyNMF (Turftopic)
- **Library:** Turftopic
- **Approach:** Keyword-focused NMF variant
- **Strengths:** Interpretable keywords, balanced performance
- **Configuration:** Turftopic KeyNMF with default settings

## Evaluation Metrics

The project uses multiple evaluation approaches:

1. **Coherence Score:** Measures semantic consistency of top words per topic
2. **Topic Diversity:** Percentage of unique words across topics
3. **Document Tightness:** Measures how well documents fit their assigned topics
4. **Outlier Rate:** Percentage of documents classified as outliers or unassigned
5. **Qualitative Review:** Manual assessment of topic quality and interpretability
6. **Runtime:** Computational efficiency and time required for model fitting

## Installation

### Requirements
- Python 3.12 (required)
- Virtual environment (recommended)

### Setup

```bash
# Clone or navigate to the project directory
cd substack-topic-modelling-comparison

# Create and activate virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode
pip install -e .

# Download required spaCy model (for tokenisation)
python -m spacy download en_core_web_sm

# [Optional] Install Jupyter kernel for notebook support
python -m ipykernel install --user --name topic-modelling

# [Optional] Install Streamlit app dependencies
pip install -r app/requirements.txt
```

## Usage

### Jupyter Notebooks

Explore the analysis step-by-step in notebooks:

```bash
jupyter notebook
```

**Notebooks**:
- `notebooks/00_data_exploration.ipynb` - Dataset overview, cleaning steps, corpus statistics
- `notebooks/01_models_default_settings.ipynb` - Train all models, compute metrics, compare results at default settings

### Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app/Home.py
```

The dashboard provides:
- Project overview and progress metrics
- Key insights from initial analysis
- Comparative model performance
- Interactive visualizations
- Navigation to detailed analysis pages

## Results

Key findings from initial analysis:
- Data cleaning removes approximately 19% of the dataset (duplicates, non-English, insufficient text)
- Modelling conducted on random sample of approximately 10% for computational efficiency
- Comparative metrics available in `artefacts/` for quick access
- Models ranked by coherence, diversity, document tightness, outlier rate and interpretability

For detailed results, refer to notebooks and dashboard.

## Data Privacy & Storage

- **Raw data** (`data/raw/`) folder is committed but contents are not (see `.gitignore`)
- **Cleaned data** (`data/interim/`) folder is committed but contents are not (see `.gitignore`)
- **Artefacts** (`artefacts/`) folder and pre-computed results (CSV, JSON) are committed for dashboard deployment

## Roadmap

### Completed ✅
- Exploratory data analysis and corpus cleaning
- Multi-model baseline comparison (TF-IDF + NMF, LDA, BERTopic, KeyNMF)
- Evaluation metrics covering coherence, topic diversity, document tightness, and outlier rate
- Interactive Streamlit dashboard for exploring results

### Planned 📋
- Hyperparameter tuning across selected models
- Multi-seed stability and variance benchmarks
- Topic stability metrics
- Qualitative topic review and visualisations



## Citation

### Dataset Citation
Substack Articles & Newsletter Metadata Dataset, by Rejaul Karim, Mendeley Data, Version 1, published 29 September 2026. DOI: [10.17632/btvzfpd33f.1](https://doi.org/10.17632/btvzfpd33f.1)

The dataset is licensed under CC BY 4.0, which requires attribution.

**Access the dataset:** [Kaggle Substack Dataset](https://www.kaggle.com/datasets/reja37/substack-newsletter-articles-and-metadata-3-5m/data)

### Project Citation
If you use this project in research, please cite:

```bibtex
@project{substack_topic_modelling_2026,
  title={Substack Topic Modelling Comparison},
  author={xaognhauinas},
  year={2026},
  url={https://github.com/xaognhauinas/substack-topic-modelling-comparison}
}
```

## Troubleshooting

### SpaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Kernel Not Found in Jupyter
```bash
python -m ipykernel install --user --name topic-modelling
```

---

**Last Updated**: November 2026