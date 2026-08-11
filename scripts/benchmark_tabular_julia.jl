# ==============================================================================
# Benchmark Pipeline Tabular - Julia
# ==============================================================================
# Este script mide:
# 1. Tiempo de importación de paquetes (Cold start / TTFX)
# 2. Carga y preprocesado de datos (Warm-up vs Ejecución)
# 3. Ajuste del modelo MLJ (Logistic Classifier / MLJLinearModels)
# 4. Evaluación de métricas y alocación de memoria
# ==============================================================================

using Dates

# 1. Medición de Cold Start (Tiempo de carga de librerías)
t_start_import = time()
using CSV, DataFrames, DataFramesMeta, MLJ, MLJLinearModels, Statistics, JSON, Random
t_import = time() - t_start_import

println("--> [Julia] Librerías cargadas en: $(round(t_import, digits=3)) s")

# 2. Generación o Ingesta de Datos Sintéticos para el Benchmark
function generate_dataset(n_rows=100_000)
    rng = MersenneTwister(42)
    return DataFrame(
        age = rand(rng, 18:80, n_rows),
        income = rand(rng, 15000.0:120000.0, n_rows),
        credit_score = rand(rng, 300:850, n_rows),
        debt_ratio = rand(rng, n_rows) .* 0.8,
        education = rand(rng, ["secundaria", "grado", "master", "doctorado"], n_rows),
        target = rand(rng, [0, 1], n_rows)
    )
end

data_path = joinpath(@__DIR__, "..", "data", "benchmark_tabular.csv")
if !isfile(data_path)
    mkpath(dirname(data_path))
    df_raw = generate_dataset(100_000)
    CSV.write(data_path, df_raw)
end

# 3. Medición de Carga y Preprocesado de Datos
t_start_prep = time()
alloc_prep = @allocated begin
    df = CSV.read(data_path, DataFrame)
    
    # Transformaciones de ingeniería de características
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
    (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, multi=true, shuffle=true, rng=42)
end
t_prep = time() - t_start_prep

println("--> [Julia] Carga y preprocesado completado en: $(round(t_prep*1000, digits=2)) ms (Memoria alocada: $(round(alloc_prep/1e6, digits=2)) MB)")

# 4. Pipeline MLJ y Entrenamiento
RegresionLogistica = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
pipe = ContinuousEncoder(drop_last=true) |> Standardizer() |> RegresionLogistica(lambda=1.0)

# Warm-up (Primera ejecución para compilar JIT)
t_start_compilation = time()
mach_warm = machine(pipe, Xtrain[1:500, :], ytrain[1:500])
fit!(mach_warm, verbosity=0)
t_compilation = time() - t_start_compilation
println("--> [Julia] Compilación JIT del fit: $(round(t_compilation, digits=3)) s")

# Medición en caliente (Warm Run)
t_start_fit = time()
alloc_fit = @allocated begin
    mach = machine(pipe, Xtrain, ytrain)
    fit!(mach, verbosity=0)
end
t_fit = time() - t_start_fit

# Evaluación de Predicciones
ŷ_prob = predict(mach, Xtest)
ŷ_mode = predict_mode(mach, Xtest)
acc = accuracy(ŷ_mode, ytest)

println("--> [Julia] Ajuste final en caliente: $(round(t_fit, digits=3)) s (Accuracy: $(round(acc, digits=4)))")

# 5. Exportar Resultados del Benchmark en JSON
results = Dict(
    "language" => "Julia",
    "version" => string(VERSION),
    "import_time_s" => round(t_import, digits=4),
    "data_prep_time_ms" => round(t_prep * 1000, digits=2),
    "data_prep_alloc_mb" => round(alloc_prep / 1e6, digits=2),
    "jit_compilation_s" => round(t_compilation, digits=4),
    "model_fit_time_s" => round(t_fit, digits=4),
    "model_fit_alloc_mb" => round(alloc_fit / 1e6, digits=2),
    "test_accuracy" => round(acc, digits=4)
)

out_json = joinpath(@__DIR__, "results_julia.json")
open(out_json, "w") do f
    JSON.print(f, results, 4)
end
println("--> [Julia] Resultados guardados en: $(out_json)")
