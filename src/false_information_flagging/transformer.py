from __future__ import annotations

from typing import Any

import numpy as np


def train_transformer_model(x_train, y_train, x_test, y_test, config: dict[str, Any]) -> tuple[Any, Any, list[int], list[float]]:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Transformer experiments require the optional transformers dependencies."
        ) from exc

    transformer_config = config["model"]["transformer"]
    training_config = config.get("training", {})
    random_seed = config.get("random_seed", 42)

    model_name = transformer_config.get("pretrained_model_name", "distilbert-base-uncased")
    max_length = transformer_config.get("max_length", 256)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_dict(
        {"text": x_train.tolist(), "label": y_train.astype(int).tolist()}
    )
    test_dataset = Dataset.from_dict(
        {"text": x_test.tolist(), "label": y_test.astype(int).tolist()}
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=config.get("artifacts", {}).get("output_dir", "artifacts/transformer"),
        per_device_train_batch_size=training_config.get("batch_size", 8),
        per_device_eval_batch_size=training_config.get("batch_size", 8),
        num_train_epochs=training_config.get("epochs", 2),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=transformer_config.get("weight_decay", 0.01),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=random_seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    outputs = trainer.predict(test_dataset)
    logits = outputs.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    predictions = np.argmax(logits, axis=1)
    return model, trainer, predictions.tolist(), probabilities.tolist()
