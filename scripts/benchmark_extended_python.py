# ==============================================================================
# Benchmark Extendido de Rendimiento y Escalabilidad - Python
# ==============================================================================

import time
import tracemalloc
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_dataset(n_rows, rng_seed=42):
    np.random.seed(rng_seed)
    return pd.DataFrame({
        "age": np.random.randint(18, 81, size=n_rows),
        "income": np.random.uniform(15000.0, 120000.0, size=n_rows),
        "credit_score": np.random.randint(300, 851, size=n_rows),
        "debt_ratio": np.random.uniform(0.0, 0.8, size=n_rows),
        "education": np.random.choice(["secundaria", "grado", "master", "doctorado"], size=n_rows),
        "target": np.random.choice([0, 1], size=n_rows)
    })

def preprocess_pipeline(df):
    df["income_per_age"] = df["income"] / df["age"]
    df["high_credit"] = df["credit_score"] > 700
    df_clean = df.dropna(subset=["income"]).copy()

    X = df_clean[["age", "income", "credit_score", "debt_ratio", "education", "income_per_age", "high_credit"]]
    y = df_clean["target"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

data_sizes = [100_000, 500_000, 1_000_000, 2_000_000]
results_by_size = {}

for N in data_sizes:
    print("==================================================")
    print(f"--> [Python] Evaluando N = {N:,} filas...")
    
    df = generate_dataset(N)
    
    # Preprocesado
    tracemalloc.start()
    t_start_prep = time.perf_counter()
    X_train, X_test, y_train, y_test = preprocess_pipeline(df)
    t_prep_ms = (time.perf_counter() - t_start_prep) * 1000
    _, peak_prep = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Fit
    tracemalloc.start()
    t_start_fit = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t_fit_s = time.perf_counter() - t_start_fit
    _, peak_fit = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"    - Preprocesado: {t_prep_ms:.2f} ms ({peak_prep/1e6:.2f} MB)")
    print(f"    - Model Fit:    {t_fit_s:.4f} s ({peak_fit/1e6:.2f} MB)")
    print(f"    - Accuracy:     {acc:.4f}")

    results_by_size[str(N)] = {
        "n_rows": N,
        "prep_time_ms": round(t_prep_ms, 2),
        "prep_alloc_mb": round(peak_prep / 1e6, 2),
        "fit_time_s": round(t_fit_s, 4),
        "fit_alloc_mb": round(peak_fit / 1e6, 2),
        "accuracy": round(acc, 4)
    }

out_json = os.path.join(os.path.dirname(__file__), "results_extended_python.json")
with open(out_json, "w") as f:
    json.dump(results_by_size, f, indent=4)

print(f"\n--> [Python] Benchmark extendido guardado en: {out_json}")
