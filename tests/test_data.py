from pathlib import Path

import pandas as pd

from false_information_flagging.data import load_and_prepare_dataset


def test_load_and_prepare_dataset_merges_title_and_assigns_labels(tmp_path: Path):
    true_path = tmp_path / "True.csv"
    fake_path = tmp_path / "Fake.csv"

    pd.DataFrame(
        [{"title": "Trusted", "text": "Verified content", "subject": "politics", "date": "2020-01-01"}]
    ).to_csv(true_path, index=False)
    pd.DataFrame(
        [{"title": "False", "text": "Made up claim", "subject": "politics", "date": "2020-01-01"}]
    ).to_csv(fake_path, index=False)

    config = {
        "dataset": {
            "true_path": str(true_path),
            "fake_path": str(fake_path),
            "text_column": "text",
            "title_column": "title",
            "merge_title": True,
            "drop_columns": ["subject", "date"],
            "label_column": "label",
        },
        "preprocessing": {
            "lowercase": True,
            "strip_html": True,
            "strip_urls": True,
            "remove_bracketed_text": True,
            "remove_stopwords": False,
            "remove_punctuation": False,
        },
    }

    df = load_and_prepare_dataset(config, tmp_path / "config.yaml")
    assert set(df["label"]) == {0, 1}
    assert "title" in df.columns
    assert "subject" not in df.columns
    assert any("trusted" in text for text in df["text"])
