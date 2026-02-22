import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def train_model():

    # ==============================
    # Configuración de rutas
    # ==============================

    GOLD_TRAIN_PATH = "data/gold/train.parquet"
    MODEL_PATH = "models/local/"
    ARTIFACTS_PATH = "artifacts/"

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    print("Cargando dataset Gold...")

    df = pd.read_parquet(GOLD_TRAIN_PATH)

    TARGET = "pago_30d"

    X = df.drop(columns=[TARGET]).values
    y = df[TARGET].values

    # ==============================
    # Definición del modelo
    # ==============================

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    print("Entrenando modelo...")

    history = model.fit(
        X,
        y,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    # ==============================
    # Evaluación final
    # ==============================

    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    metrics = {
        "AUC": float(auc),
        "Precision": float(precision),
        "Recall": float(recall)
    }

    print("Resultados del modelo:")
    print(metrics)

    # ==============================
    # Guardar modelo y métricas
    # ==============================

    model.save(os.path.join(MODEL_PATH, "model.keras"))

    with open(os.path.join(ARTIFACTS_PATH, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("✔ Modelo guardado en models/local/")
    print("✔ Métricas guardadas en artifacts/")