# ==============================================================================
# Benchmark Extendido de Rendimiento y Escalabilidad - Julia
# ==============================================================================

using Dates, Random, JSON
using CSV, DataFrames, DataFramesMeta, MLJ, MLJLinearModels, Statistics

# 1. Medición de Cold Start (Carga de librerías)
t_start_import = time()
# (Librerías ya importadas en el script)
t_import = time() - t_start_import

# 2. Generador de Datos
function generate_dataset(n_rows, rng_seed=42)
    rng = MersenneTwister(rng_seed)
    return DataFrame(
        age = rand(rng, 18:80, n_rows),
        income = rand(rng, 15000.0:120000.0, n_rows),
        credit_score = rand(rng, 300:850, n_rows),
        debt_ratio = rand(rng, n_rows) .* 0.8,
        education = rand(rng, ["secundaria", "grado", "master", "doctorado"], n_rows),
        target = rand(rng, [0, 1], n_rows)
    )
end

# 3. Función de Preprocesado aislada
function preprocess_pipeline(df::DataFrame)
    df_clean = @chain df begin
        @transform(
            :income_per_age = :income ./ :age,
            :high_credit = :credit_score .> 700
        )
        @rsubset(!ismissing(:income))
    end
    
    df_clean.target = coerce(df_clean.target, Multiclass)
    df_clean.education = coerce(df_clean.education, Multiclass)
    df_clean.high_credit = coerce(df_clean.high_credit, Multiclass)
    df_clean = coerce(df_clean, Count => Continuous)
    
    y, X = unpack(df_clean, ==(:target))
    return partition((X, y), 0.8, multi=true, shuffle=true, rng=42)
end

# 4. WARM-UP: Compilar funciones de preprocesado y fit con dataset diminuto (100 filas)
println("--> [Julia] Ejecutando Warm-Up de compilación JIT...")
df_warm = generate_dataset(100)
(Xtr_w, Xte_w), (ytr_w, yte_w) = preprocess_pipeline(df_warm)

RegresionLogistica = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
pipe = ContinuousEncoder(drop_last=true) |> Standardizer() |> RegresionLogistica(lambda=1.0)
mach_warm = machine(pipe, Xtr_w, ytr_w)
fit!(mach_warm, verbosity=0)
predict(mach_warm, Xte_w)
println("--> [Julia] Warm-Up completado (Código 100% compilado JIT).\n")

# 5. Benchmarking de Escalabilidad para Varios Tamaños (N = 100k, 500k, 1M, 2M)
data_sizes = [100_000, 500_000, 1_000_000, 2_000_000]
results_by_size = Dict()

for N in data_sizes
    println("==================================================")
    println("--> [Julia] Evaluando N = $(N) filas...")
    
    # Generar dataset
    df = generate_dataset(N)
    
    # Medir Preprocesado en Caliente
    t_start_prep = time()
    alloc_prep = @allocated begin
        (Xtrain, Xtest), (ytrain, ytest) = preprocess_pipeline(df)
    end
    t_prep_ms = (time() - t_start_prep) * 1000
    
    # Medir Entrenamiento (Fit) en Caliente
    t_start_fit = time()
    alloc_fit = @allocated begin
        mach = machine(pipe, Xtrain, ytrain)
        fit!(mach, verbosity=0)
    end
    t_fit_s = time() - t_start_fit
    
    # Evaluar Precisión
    ŷ_mode = predict_mode(mach, Xtest)
    acc = accuracy(ŷ_mode, ytest)
    
    println("    - Preprocesado: $(round(t_prep_ms, digits=2)) ms ($(round(alloc_prep/1e6, digits=2)) MB)")
    println("    - Model Fit:    $(round(t_fit_s, digits=4)) s ($(round(alloc_fit/1e6, digits=2)) MB)")
    println("    - Accuracy:     $(round(acc, digits=4))")
    
    results_by_size[string(N)] = Dict(
        "n_rows" => N,
        "prep_time_ms" => round(t_prep_ms, digits=2),
        "prep_alloc_mb" => round(alloc_prep / 1e6, digits=2),
        "fit_time_s" => round(t_fit_s, digits=4),
        "fit_alloc_mb" => round(alloc_fit / 1e6, digits=2),
        "accuracy" => round(acc, digits=4)
    )
end

# 6. Guardar Resultados Extendidos
out_json = joinpath(@__DIR__, "results_extended_julia.json")
open(out_json, "w") do f
    JSON.print(f, results_by_size, 4)
end
println("\n--> [Julia] Benchmark extendido guardado en: $(out_json)")
