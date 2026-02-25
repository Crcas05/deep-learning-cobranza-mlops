import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from google.cloud import bigquery


def batch_predict():

    # ==============================
    # RUTAS CLOUD
    # ==============================

    MODEL_PATH = "gs://mlops-cobranza-artifacts-34614/model/model.keras"
    SILVER_PATH = "gs://mlops-cobranza-artifacts-34614/model/silver/cobranza_clean.parquet"

    print("Cargando modelo desde GCS...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Cargando datos Silver desde GCS...")
    df = pd.read_parquet(SILVER_PATH)

    TARGET = "pago_30d"

    df_scoring = df.copy()

    # ==============================
    # Preparar features correctamente
    # ==============================

    if TARGET in df.columns:
        print("Columna target encontrada. Eliminándola para inferencia.")
        X = df.drop(columns=[TARGET])
    else:
        print("Columna target NO encontrada. Usando todas las columnas como features.")
        X = df

    # Asegurar que sea numpy array
    X = X.values

    # ==============================
    # Predicción
    # ==============================

    print("Generando predicciones...")
    probs = model.predict(X)

    # Si viene como array 2D (ej: [[0.8], [0.3]])
    if len(probs.shape) > 1:
        probs = probs.flatten()

    df_scoring["score_probabilidad"] = probs.astype(float)

    # ==============================
    # Segmentación
    # ==============================

    def asignar_prioridad(p):
        if p >= 0.7:
            return "Alta"
        elif p >= 0.4:
            return "Media"
        else:
            return "Baja"

    df_scoring["prioridad"] = df_scoring["score_probabilidad"].apply(asignar_prioridad)

    # ==============================
    # Guardar en BigQuery
    # ==============================

    print("Guardando predicciones en BigQuery...")

    client = bigquery.Client()

    table_id = "solid-league-440122-i6.mlops_cobranza.predicciones_batch"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND"  # agrega registros
    )

    job = client.load_table_from_dataframe(
        df_scoring,
        table_id,
        job_config=job_config
    )

    job.result()

    print("✔ Predicciones guardadas en BigQuery correctamente.")


if __name__ == "__main__":
    batch_predict()