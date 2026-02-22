import os
import pandas as pd
from sklearn.model_selection import train_test_split

SILVER_PATH = "data/silver/cobranza_clean.parquet"
GOLD_PATH = "data/gold/"

os.makedirs(GOLD_PATH, exist_ok=True)

print("Cargando datos Silver...")
df = pd.read_parquet(SILVER_PATH)

TARGET = "pago_30d"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df[TARGET] = y_train.values

test_df = X_test.copy()
test_df[TARGET] = y_test.values

train_df.to_parquet(os.path.join(GOLD_PATH, "train.parquet"), index=False)
test_df.to_parquet(os.path.join(GOLD_PATH, "test.parquet"), index=False)

print("✔ Gold generado correctamente.")