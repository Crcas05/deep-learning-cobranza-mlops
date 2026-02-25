import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def train_model():

    # ==============================
    # RUTAS CLOUD (VERTEX)
    # ==============================

    GOLD_TRAIN_PATH = "gs://mlops-cobranza-artifacts-34614/gold/train.parquet"

    # Estas variables las inyecta Vertex automáticamente
    MODEL_DIR = os.environ["AIP_MODEL_DIR"]
    ARTIFACTS_DIR = os.environ["AIP_OUTPUT_DIR"]

    print("MODEL_DIR:", MODEL_DIR)
    print("ARTIFACTS_DIR:", ARTIFACTS_DIR)

    print("Cargando dataset Gold desde GCS...")
    df = pd.read_parquet(GOLD_TRAIN_PATH)

    TARGET = "pago_30d"

    X = df.drop(columns=[TARGET]).values
    y = df[TARGET].values

    # ==============================
    # MODELO
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

    model.fit(
        X,
        y,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    # ==============================
    # MÉTRICAS
    # ==============================

    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "AUC": float(roc_auc_score(y, y_pred_proba)),
        "Precision": float(precision_score(y, y_pred)),
        "Recall": float(recall_score(y, y_pred))
    }

    print("Resultados:", metrics)

    # ==============================
    # GUARDADO CORRECTO EN VERTEX
    # ==============================

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Guardar modelo en directorio especial de Vertex
    model_path = os.path.join(MODEL_DIR, "model.keras")
    model.save(model_path)

    print(f"Modelo guardado en: {model_path}")

    # Guardar métricas
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    train_model()