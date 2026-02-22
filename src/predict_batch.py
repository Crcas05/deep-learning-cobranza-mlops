import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# ==============================
# Rutas
# ==============================

MODEL_PATH = "models/local/model.keras"
SCALER_PATH = "artifacts/scaler.pkl"
SILVER_PATH = "data/silver/cobranza_clean.parquet"
OUTPUT_PATH = "artifacts/scoring_results.parquet"

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Cargando scaler...")
scaler = joblib.load(SCALER_PATH)

print("Cargando datos Silver...")
df = pd.read_parquet(SILVER_PATH)

TARGET = "pago_30d"

# Guardamos copia original para scoring
df_scoring = df.copy()

# Separar features
X = df.drop(columns=[TARGET]).values

# Escalar
X_scaled = scaler.transform(X)

# ==============================
# Predicción
# ==============================

print("Generando predicciones...")
probs = model.predict(X_scaled)

df_scoring["score_probabilidad"] = probs
df_scoring["score_probabilidad"] = df_scoring["score_probabilidad"].astype(float)

# ==============================
# Segmentación de prioridad
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
# Guardar resultado
# ==============================

df_scoring.to_parquet(OUTPUT_PATH, index=False)

print("✔ Scoring batch generado correctamente.")
print(f"✔ Archivo guardado en {OUTPUT_PATH}")