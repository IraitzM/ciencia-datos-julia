# ==============================================================================
# Benchmark Extendido Deep Learning / Optimización Iterativa - Julia (Flux.jl)
# ==============================================================================
# Problema: Regresión no lineal sobre datos sintéticos.
# Red: MLP de 3 capas (Dense → ReLU → Dense → ReLU → Dense)
# Métricas por tamaño (100k, 500k, 1M filas):
#   - Tiempo de Warm-Up (JIT de Flux + Zygote)
#   - Tiempo por Época en Caliente
#   - Tiempo hasta Convergencia (Loss ≤ umbral ε)
#   - Memoria alocada por época
# ==============================================================================

using Random, JSON, Statistics

println("--> [Julia] Cargando librerías...")
t_load = @elapsed begin
    using Flux
end
println("--> [Julia] Flux + Zygote cargados en: $(round(t_load, digits=3)) s")

# ==============================================================================
# 1. Generador de Datos Sintéticos (Regresión no lineal: y = sin(x1) + x2^2 + ε)
# ==============================================================================
function generate_regression_data(n_rows::Int, n_features::Int=10, rng_seed::Int=42)
    rng = MersenneTwister(rng_seed)
    X = randn(rng, Float32, n_features, n_rows)
    y = sin.(X[1:1, :]) .+ X[2:2, :].^2 .+ 0.1f0 .* randn(rng, Float32, 1, n_rows)
    return X, y
end

# ==============================================================================
# 2. Definición del Modelo MLP
# ==============================================================================
function build_model(n_features::Int=10)
    return Chain(
        Dense(n_features => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => 1)
    )
end

loss_fn(model, X, y) = Flux.mse(model(X), y)

# ==============================================================================
# 3. Warm-Up JIT con dataset diminuto (50 filas, 5 épocas)
# ==============================================================================
println("\n--> [Julia] Warm-Up de compilación JIT (Flux + Zygote AD)...")
t_jit = @elapsed begin
    Xw, yw = generate_regression_data(50)
    model_w = build_model()
    opt_w = Flux.setup(Adam(1e-3), model_w)
    for _ in 1:5
        grads = Flux.gradient(m -> loss_fn(m, Xw, yw), model_w)
        Flux.update!(opt_w, model_w, grads[1])
    end
end
println("--> [Julia] Warm-Up completado en: $(round(t_jit, digits=3)) s\n")

# ==============================================================================
# 4. Benchmark por Tamaño con métricas completas
# ==============================================================================
LOSS_TARGET = 0.15f0    # umbral de convergencia (MSE)
N_EPOCHS    = 50        # épocas máximas
BATCH_SIZE  = 512
data_sizes  = [100_000, 500_000, 1_000_000]
results     = Dict()

for N in data_sizes
    println("=" ^ 50)
    println("--> [Julia] Evaluando N = $N filas...")

    X, y = generate_regression_data(N)

    # --- Tiempo por Época en Caliente ---
    model = build_model()
    opt   = Flux.setup(Adam(1e-3), model)
    loader = Flux.DataLoader((X, y), batchsize=BATCH_SIZE, shuffle=true)

    epoch_times = Float64[]
    epoch_losses = Float64[]
    alloc_per_epoch = Int64[]
    converged_at = N_EPOCHS + 1  # sin convergencia por defecto

    for epoch in 1:N_EPOCHS
        t_ep = @elapsed begin
            alloc_ep = @allocated begin
                for (Xb, yb) in loader
                    grads = Flux.gradient(m -> loss_fn(m, Xb, yb), model)
                    Flux.update!(opt, model, grads[1])
                end
            end
            push!(alloc_per_epoch, alloc_ep)
        end

        current_loss = loss_fn(model, X, y)
        push!(epoch_times, t_ep)
        push!(epoch_losses, Float64(current_loss))

        if current_loss <= LOSS_TARGET && converged_at > N_EPOCHS
            converged_at = epoch
        end
    end

    final_loss = epoch_losses[end]
    t_per_epoch_ms = mean(epoch_times[2:end]) * 1000  # Excluimos época 1 (puede incluir compilación extra)
    t_convergence = converged_at <= N_EPOCHS ? sum(epoch_times[1:converged_at]) : NaN
    alloc_mb = mean(alloc_per_epoch[2:end]) / 1e6

    println("    - Tiempo por Época (Warm):  $(round(t_per_epoch_ms, digits=2)) ms")
    println("    - Épocas hasta Convergencia: $converged_at / $N_EPOCHS  (target loss ≤ $LOSS_TARGET)")
    println("    - Tiempo hasta Convergencia: $(isnan(t_convergence) ? "No convergió" : round(t_convergence, digits=3)) s")
    println("    - Loss Final (época $N_EPOCHS):  $(round(final_loss, digits=5))")
    println("    - Memoria por Época (Warm):  $(round(alloc_mb, digits=2)) MB")

    results[string(N)] = Dict(
        "n_rows"                => N,
        "epoch_time_ms"         => round(t_per_epoch_ms, digits=2),
        "convergence_epoch"     => converged_at <= N_EPOCHS ? converged_at : -1,
        "convergence_time_s"    => isnan(t_convergence) ? -1.0 : round(t_convergence, digits=4),
        "final_loss"            => round(final_loss, digits=6),
        "alloc_per_epoch_mb"    => round(alloc_mb, digits=2)
    )
end

# ==============================================================================
# 5. Guardar Resultados
# ==============================================================================
out_json = joinpath(@__DIR__, "results_dl_julia.json")
open(out_json, "w") do f
    JSON.print(f, results, 4)
end
println("\n--> [Julia] Resultados Deep Learning guardados en: $out_json")
