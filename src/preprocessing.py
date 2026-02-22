import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# Rutas
# ==============================

BRONZE_PATH = "data/bronze/cobranza_sintetica.csv"
SILVER_PATH = "data/silver/"
ARTIFACTS_PATH = "artifacts/"

os.makedirs(SILVER_PATH, exist_ok=True)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

print("Cargando datos Bronze...")
df = pd.read_csv(BRONZE_PATH)

# ==============================
# Limpieza básica
# ==============================

df = df.drop_duplicates()
df = df.dropna()

# ==============================
# Feature Engineering
# ==============================

df["ratio_pago_mora"] = df["porcentaje_pago_historico"] / (df["dias_mora_actual"] + 1)
df["ingreso_deuda_ratio"] = df["ingresos_estimados"] / (df["monto_deuda"] + 1)

TARGET = "pago_30d"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==============================
# Escalado
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar scaler
joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.pkl"))

# Reconstruir dataframe escalado
df_silver = pd.DataFrame(X_scaled, columns=X.columns)
df_silver[TARGET] = y.values

# ==============================
# Guardar Silver en Parquet
# ==============================

df_silver.to_parquet(
    os.path.join(SILVER_PATH, "cobranza_clean.parquet"),
    index=False
)

print("✔ Silver generado correctamente.")
print("✔ Scaler guardado en artifacts/")