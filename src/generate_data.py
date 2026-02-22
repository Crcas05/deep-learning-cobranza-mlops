import numpy as np
import pandas as pd
import os

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
N = 80000
np.random.seed(42)

# -------------------------------
# GENERACIÓN DE VARIABLES
# -------------------------------

cliente_id = np.arange(1, N + 1)

# Mora actual entre 1 y 30 días
dias_mora_actual = np.random.randint(1, 31, N)

# Ingresos estimados (distribución normal truncada)
ingresos_estimados = np.clip(
    np.random.normal(loc=2500, scale=800, size=N), 800, 8000
)

# Monto de deuda correlacionado con ingresos
monto_deuda = ingresos_estimados * np.random.uniform(0.5, 2.5, N)

# Historial de pago (0 a 1)
porcentaje_pago_historico = np.clip(
    np.random.normal(loc=0.7, scale=0.15, size=N), 0, 1
)

# Número de atrasos previos
numero_atrasos_previos = np.random.poisson(lam=2, size=N)

# Antigüedad del cliente (meses)
antiguedad_cliente = np.random.randint(6, 120, N)

# Ratio deuda/ingreso
ratio_deuda_ingreso = monto_deuda / ingresos_estimados

# -------------------------------
# MODELO PROBABILÍSTICO NO LINEAL
# -------------------------------

logit = (
    -0.08 * dias_mora_actual
    + 2.5 * porcentaje_pago_historico
    - 0.5 * ratio_deuda_ingreso
    - 0.2 * numero_atrasos_previos
    + 0.01 * antiguedad_cliente
)

probabilidad_pago = 1 / (1 + np.exp(-logit))

pago_30d = np.random.binomial(1, probabilidad_pago)

# -------------------------------
# CREAR DATAFRAME
# -------------------------------

df = pd.DataFrame({
    "cliente_id": cliente_id,
    "dias_mora_actual": dias_mora_actual,
    "ingresos_estimados": ingresos_estimados,
    "monto_deuda": monto_deuda,
    "ratio_deuda_ingreso": ratio_deuda_ingreso,
    "porcentaje_pago_historico": porcentaje_pago_historico,
    "numero_atrasos_previos": numero_atrasos_previos,
    "antiguedad_cliente": antiguedad_cliente,
    "pago_30d": pago_30d
})

# -------------------------------
# GUARDAR EN BRONZE
# -------------------------------

output_path = "data/bronze/cobranza_sintetica.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_csv(output_path, index=False)

print("Dataset generado correctamente en:", output_path)
print(df.head())