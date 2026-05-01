# Experiment Strategy

## Recommended Evaluation Order

1. Train `classical_baseline.yaml` first to establish a fast and explainable benchmark.
2. Train `neural_bilstm.yaml` to compare recurrent sequence modeling against sparse-text baselines.
3. Train `transformer_distilbert.yaml` to measure gains from contextual language modeling.

## What To Report

- dataset size after cleaning
- train/test split proportions
- accuracy, precision, recall, F1, ROC-AUC
- confusion matrix
- failure cases or misclassified examples
- training cost and runtime notes

## Strong Research Questions

- How much performance lift does contextual modeling provide over TF-IDF baselines?
- Does removing stopwords help classical models more than neural or transformer models?
- Does concatenating `title` and `text` outperform using body text alone?
- Which models are most stable when article length varies?

## Recommended Error Analysis

- Compare false positives versus false negatives.
- Slice mistakes by article subject if metadata is retained in a side table.
- Inspect whether models overfit lexical cues such as sensational verbs or named entities.

