# ==============================================================================
# Benchmark Extendido Deep Learning / Optimización Iterativa - Python (PyTorch)
# ==============================================================================
# Problema: Regresión no lineal sobre datos sintéticos.
# Red: MLP de 3 capas equivalente a la versión Julia (Flux.jl)
# Métricas por tamaño (100k, 500k, 1M filas):
#   - Tiempo por Época en Caliente
#   - Tiempo hasta Convergencia (Loss ≤ umbral ε)
#   - Memoria alocada por época (via tracemalloc)
# ==============================================================================

import time
import json
import os
import tracemalloc
import numpy as np

print("--> [Python] Cargando librerías...")
t_load = time.perf_counter()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
t_load = time.perf_counter() - t_load
print(f"--> [Python] torch cargado en: {t_load:.3f} s")

# ==============================================================================
# 1. Generador de Datos Sintéticos (idéntico al de Julia)
# ==============================================================================
def generate_regression_data(n_rows, n_features=10, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    y = (np.sin(X[:, 0:1]) + X[:, 1:2] ** 2 + 0.1 * rng.standard_normal((n_rows, 1))).astype(np.float32)
    return X, y

# ==============================================================================
# 2. Definición del Modelo MLP (equivalente a Flux.jl)
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, n_features=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

loss_fn = nn.MSELoss()

# ==============================================================================
# 3. Benchmark por Tamaño
# ==============================================================================
LOSS_TARGET = 0.15
N_EPOCHS    = 50
BATCH_SIZE  = 512
data_sizes  = [100_000, 500_000, 1_000_000]
results     = {}

for N in data_sizes:
    print("=" * 50)
    print(f"--> [Python] Evaluando N = {N:,} filas...")

    X_np, y_np = generate_regression_data(N)
    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_times  = []
    epoch_losses = []
    alloc_mb_list = []
    converged_at = N_EPOCHS + 1

    for epoch in range(1, N_EPOCHS + 1):
        tracemalloc.start()
        t_start = time.perf_counter()

        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        t_ep = time.perf_counter() - t_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Eval loss sobre todo el dataset
        model.eval()
        with torch.no_grad():
            current_loss = loss_fn(model(X_t), y_t).item()

        epoch_times.append(t_ep)
        epoch_losses.append(current_loss)
        alloc_mb_list.append(peak_mem / 1e6)

        if current_loss <= LOSS_TARGET and converged_at > N_EPOCHS:
            converged_at = epoch

    final_loss     = epoch_losses[-1]
    # Excluimos época 1 (puede incluir inicializaciones del DataLoader)
    t_per_epoch_ms = np.mean(epoch_times[1:]) * 1000
    t_convergence  = sum(epoch_times[:converged_at]) if converged_at <= N_EPOCHS else float("nan")
    alloc_mb       = np.mean(alloc_mb_list[1:])

    print(f"    - Tiempo por Época (Warm):   {t_per_epoch_ms:.2f} ms")
    print(f"    - Épocas hasta Convergencia: {converged_at} / {N_EPOCHS}  (target loss ≤ {LOSS_TARGET})")
    print(f"    - Tiempo hasta Convergencia: {'No convergió' if np.isnan(t_convergence) else f'{t_convergence:.3f} s'}")
    print(f"    - Loss Final (época {N_EPOCHS}):   {final_loss:.5f}")
    print(f"    - Memoria por Época (Warm):  {alloc_mb:.2f} MB")

    results[str(N)] = {
        "n_rows":              N,
        "epoch_time_ms":       round(t_per_epoch_ms, 2),
        "convergence_epoch":   converged_at if converged_at <= N_EPOCHS else -1,
        "convergence_time_s":  -1.0 if np.isnan(t_convergence) else round(t_convergence, 4),
        "final_loss":          round(final_loss, 6),
        "alloc_per_epoch_mb":  round(alloc_mb, 2)
    }

# ==============================================================================
# 4. Guardar Resultados
# ==============================================================================
out_json = os.path.join(os.path.dirname(__file__), "results_dl_python.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n--> [Python] Resultados Deep Learning guardados en: {out_json}")
