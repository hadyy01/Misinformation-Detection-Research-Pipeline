# False Information Flagging Algorithm

This repository turns an exploratory notebook into a research-oriented machine learning project for misinformation detection. The codebase is structured to support reproducible experiments, stronger engineering practices, and a cleaner story for GitHub, interviews, and resume discussions.

## Project Highlights

- Comparative NLP research workflow spanning linear, neural, and transformer baselines.
- Config-driven experimentation for reproducibility and clearer model comparisons.
- Research artifacts saved automatically for reporting, portfolio screenshots, and result tracking.
- Repository structure designed to read well for hiring managers, collaborators, and academic-style reviews.

## Why This Repository Is Stronger Than the Original Notebook

- Separates data loading, preprocessing, modeling, evaluation, and reporting into reusable Python modules.
- Fixes the original workflow issue where TF-IDF features were mixed with sequence-model layers in the same training path.
- Supports three clear experiment families:
  - `classical`: TF-IDF + linear baseline models for fast, reproducible comparisons.
  - `neural`: sequence-based BiLSTM experiments built on raw text rather than sparse TF-IDF matrices.
  - `transformer`: fine-tuning-ready transformer configuration for stronger modern NLP baselines.
- Uses config-driven runs so experiment settings are explicit, repeatable, and easy to compare.
- Preserves the original notebook as a research artifact in [notebooks/legacy_research_notebook.ipynb](notebooks/legacy_research_notebook.ipynb).

## Repository Layout

```text
.
|-- configs/                     # Experiment configs
|-- data/
|   `-- README.md               # Dataset placement and schema notes
|-- docs/
|   |-- architecture.md         # System design and workflow
|   `-- portfolio_positioning.md
|-- notebooks/
|   |-- legacy_research_notebook.ipynb
|   `-- README.md
|-- reports/
|   `-- model_card_template.md
|-- scripts/
|   `-- train.py                # Training entrypoint
|-- src/false_information_flagging/
|   |-- config.py
|   |-- data.py
|   |-- classical.py
|   |-- neural.py
|   |-- transformer.py
|   |-- metrics.py
|   |-- pipeline.py
|   `-- preprocessing.py
|-- tests/
|   |-- test_data.py
|   `-- test_preprocessing.py
`-- artifacts/                  # Run outputs, ignored by git
```

## Recommended Research Framing

Position this project as a comparative misinformation-detection study rather than only a single-model notebook:

- Problem: binary classification of credible vs false news articles.
- Research question: which text representation and modeling family provides the best balance of performance, interpretability, and reproducibility?
- Baselines: Logistic Regression, Linear SVM, Multinomial Naive Bayes on TF-IDF features.
- Neural track: BiLSTM over tokenized text sequences.
- Transformer track: `distilbert-base-uncased` fine-tuning path for stronger contextual language modeling.
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and experiment summaries saved to disk.

## Quick Start

1. Create a virtual environment and install dependencies.
2. Place the source CSVs at `data/raw/True.csv` and `data/raw/Fake.csv`.
3. Run a classical baseline:

```bash
python scripts/train.py --config configs/classical_baseline.yaml
```

4. Run the neural experiment after installing the optional deep learning dependency:

```bash
pip install ".[deep-learning]"
python scripts/train.py --config configs/neural_bilstm.yaml
```

5. Run the transformer experiment after installing the transformer dependencies:

```bash
pip install ".[transformers]"
python scripts/train.py --config configs/transformer_distilbert.yaml
```

Artifacts are written to the configured output directory inside `artifacts/`.

## Suggested Next Research Extensions

- Add cross-validation and per-topic robustness checks.
- Compare TF-IDF with transformer embeddings.
- Measure calibration and threshold sensitivity.
- Add error analysis slices for political topic, article length, and source style.
- Track experiments with MLflow or Weights & Biases.
