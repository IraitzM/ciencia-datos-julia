# Ciencia de datos con Julia

<img src="images/cover.png" alt="cover" width="200"/>

Libro guía sobre cómo usar este lenguaje de programación para ciencia de datos.

📖 **Léelo en línea: <https://iraitzm.github.io/ciencia-datos-julia>**

## Requisitos

| Herramienta | Versión |
|---|---|
| [Julia](https://julialang.org/downloads/) | 1.11 o superior (probado con 1.12) |
| [Quarto](https://quarto.org/docs/download/) | **1.7 o superior** |

> [!IMPORTANT]
> Quarto por debajo de 1.7 fija `QuartoNotebookRunner =0.11.6`, que no es
> compatible con Julia 1.12 y falla al precompilar. Si usas Julia 1.12,
> necesitas Quarto 1.7+.

## Construir el libro

```sh
git clone https://github.com/IraitzM/ciencia-datos-julia.git
cd ciencia-datos-julia

# Instala exactamente las versiones fijadas en Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'

quarto preview   # o `quarto render` para generar _book/
```

No hace falta instalar los paquetes uno a uno: `Project.toml` y `Manifest.toml`
están versionados precisamente para que el entorno sea reproducible.

## Estructura

| Ruta | Contenido |
|---|---|
| `index.qmd`, `intro.qmd` | Prólogo e introducción a Julia |
| `parts/firststeps/` | Sintaxis, flujos, módulos y sistema de ficheros |
| `parts/dataframes/` | Carga de datos, análisis preliminar y exploratorio |
| `data/` | Conjuntos de datos usados por los capítulos |
| `_freeze/` | Resultados congelados de la ejecución (ver más abajo) |

## Sobre la reproducibilidad

- `Project.toml` / `Manifest.toml` fijan el entorno Julia completo.
- `execute: freeze: auto` guarda en `_freeze/` el resultado de ejecutar cada
  capítulo. Solo se reejecuta lo que cambia, y la publicación no depende de que
  las APIs externas que se usan como ejemplo estén disponibles.
- Los capítulos no modifican ficheros versionados: lo que generan (bases DuckDB,
  descargas) va a directorios temporales.
