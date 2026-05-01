from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_classical_pipeline(model_config: dict[str, Any]) -> Pipeline:
    tfidf_config = model_config.get("tfidf", {})
    estimator_config = model_config.get("estimator", {})
    model_name = model_config.get("name", "logistic_regression")

    vectorizer = TfidfVectorizer(
        max_features=tfidf_config.get("max_features", 40000),
        min_df=tfidf_config.get("min_df", 3),
        ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
    )

    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=estimator_config.get("C", 4.0),
            max_iter=estimator_config.get("max_iter", 2000),
        )
    elif model_name == "linear_svm":
        estimator = LinearSVC(C=estimator_config.get("C", 1.0))
    elif model_name == "multinomial_nb":
        estimator = MultinomialNB(alpha=estimator_config.get("alpha", 1.0))
    else:
        raise ValueError(f"Unsupported classical model: {model_name}")

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("estimator", estimator),
        ]
    )


def predict_scores(pipeline: Pipeline, x_test):
    estimator = pipeline.named_steps["estimator"]
    if hasattr(estimator, "predict_proba"):
        return pipeline.predict_proba(x_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        return pipeline.decision_function(x_test)
    return pipeline.predict(x_test)

