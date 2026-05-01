from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve_path
from .preprocessing import clean_text


def load_and_prepare_dataset(config: dict[str, Any], config_path: Path) -> pd.DataFrame:
    dataset_config = config["dataset"]
    preprocessing_config = config.get("preprocessing", {})
    base_dir = (
        config_path.parent.parent.resolve()
        if config_path.parent.name == "configs"
        else config_path.parent.resolve()
    )

    true_path = resolve_path(base_dir, dataset_config["true_path"])
    fake_path = resolve_path(base_dir, dataset_config["fake_path"])

    true_df = pd.read_csv(true_path).copy()
    fake_df = pd.read_csv(fake_path).copy()

    label_column = dataset_config.get("label_column", "label")
    true_df[label_column] = 1
    fake_df[label_column] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)

    text_column = dataset_config.get("text_column", "text")
    title_column = dataset_config.get("title_column", "title")
    merge_title = dataset_config.get("merge_title", True)
    drop_columns = dataset_config.get("drop_columns", [])

    if merge_title and title_column in df.columns:
        titles = df[title_column].fillna("")
        texts = df[text_column].fillna("")
        df[text_column] = texts.astype(str) + " " + titles.astype(str)

    for column in drop_columns:
        if column in df.columns:
            df = df.drop(columns=column)

    df[text_column] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .apply(lambda value: clean_text(value, **preprocessing_config))
    )

    df = df[df[text_column].str.len() > 0].reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    dataset_config = config["dataset"]
    split_config = config.get("split", {})

    text_column = dataset_config.get("text_column", "text")
    label_column = dataset_config.get("label_column", "label")
    random_seed = config.get("random_seed", 42)
    test_size = split_config.get("test_size", 0.2)
    stratify_enabled = split_config.get("stratify", True)

    x = df[text_column]
    y = df[label_column]

    stratify = y if stratify_enabled else None
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )
