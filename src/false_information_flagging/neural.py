from __future__ import annotations

from typing import Any


def train_neural_model(x_train, y_train, x_test, y_test, config: dict[str, Any]) -> tuple[Any, Any, list[float]]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for neural experiments. Install the optional deep-learning dependencies."
        ) from exc

    model_config = config["model"]["neural"]
    training_config = config.get("training", {})
    random_seed = config.get("random_seed", 42)

    tf.keras.utils.set_random_seed(random_seed)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=model_config.get("max_tokens", 20000),
        output_mode="int",
        output_sequence_length=model_config.get("sequence_length", 400),
    )
    vectorizer.adapt(x_train.to_numpy())

    inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(
        input_dim=model_config.get("max_tokens", 20000),
        output_dim=model_config.get("embedding_dim", 128),
        mask_zero=True,
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            model_config.get("lstm_units", 96),
            return_sequences=False,
        )
    )(x)
    x = tf.keras.layers.Dropout(model_config.get("dropout", 0.3))(x)
    x = tf.keras.layers.Dense(model_config.get("dense_units", 64), activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config["model"].get("name", "bilstm"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.get("learning_rate", 0.001)),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train.to_numpy(),
        y_train.to_numpy(),
        validation_data=(x_test.to_numpy(), y_test.to_numpy()),
        epochs=training_config.get("epochs", 5),
        batch_size=training_config.get("batch_size", 64),
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(x_test.to_numpy(), verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    return model, history, predictions.tolist(), probabilities.tolist()

