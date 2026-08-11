# ==============================================================================
# Benchmark Pipeline Tabular - Python
# ==============================================================================
# Este script mide:
# 1. Tiempo de importación de librerías (Cold start)
# 2. Carga y preprocesado de datos (Polars/Pandas + Scikit-Learn)
# 3. Entrenamiento con LogisticRegression / LightGBM
# 4. Evaluación de métricas y alocación de memoria
# ==============================================================================

import time
import tracemalloc
import os
import json
import sys

# 1. Medición de Cold Start (Tiempo de carga de librerías)
t_start_import = time.perf_counter()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
t_import = time.perf_counter() - t_start_import

print(f"--> [Python] Librerías cargadas en: {t_import:.3f} s")

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_tabular.csv")

# 2. Medición de Carga y Preprocesado de Datos
tracemalloc.start()
t_start_prep = time.perf_counter()

df = pd.read_csv(data_path)

# Ingenieria de características equivalente
df["income_per_age"] = df["income"] / df["age"]
df["high_credit"] = df["credit_score"] > 700
df_clean = df.dropna(subset=["income"]).copy()

X = df_clean[["age", "income", "credit_score", "debt_ratio", "education", "income_per_age", "high_credit"]]
y = df_clean["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

t_prep = time.perf_counter() - t_start_prep
current_prep, peak_prep = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"--> [Python] Carga y preprocesado en: {t_prep*1000:.2f} ms (Peak RAM: {peak_prep/1e6:.2f} MB)")

# 3. Pipeline de Scikit-Learn y Entrenamiento
num_cols = ["age", "income", "credit_score", "debt_ratio", "income_per_age"]
cat_cols = ["education"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Medición del Ajuste (Fit)
tracemalloc.start()
t_start_fit = time.perf_counter()

pipeline.fit(X_train, y_train)

t_fit = time.perf_counter() - t_start_fit
current_fit, peak_fit = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Evaluación de Predicciones
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"--> [Python] Ajuste del pipeline en: {t_fit:.3f} s (Accuracy: {acc:.4f})")

# 4. Exportar Resultados del Benchmark en JSON
results = {
    "language": "Python",
    "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "import_time_s": round(t_import, 4),
    "data_prep_time_ms": round(t_prep * 1000, 2),
    "data_prep_alloc_mb": round(peak_prep / 1e6, 2),
    "jit_compilation_s": 0.0,  # Python interpretado / C backend
    "model_fit_time_s": round(t_fit, 4),
    "model_fit_alloc_mb": round(peak_fit / 1e6, 2),
    "test_accuracy": round(acc, 4)
}

out_json = os.path.join(os.path.dirname(__file__), "results_python.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"--> [Python] Resultados guardados en: {out_json}")
