---
title: "Flujos en Julia"
format:
  html:
    code-fold: false
engine: julia
---

Una vez controlamos las estructuras base podremos aplicar distintas lógicas de flujo a la hora de evaluar nuestro código.

## Condición (if)

Los flujos `if` muestran un funcionamiento similar a otros lenguajes de programación.

```julia
a = 1
b = 2

if a < b
    "a < b"
elseif a > b
    "a > b"
else
    "a == b"
end
```

La sentencia debe iniciar con `if` y terminar con `end`. Internamente podemos evaluar condiciones adicionales con `elseif` o condiciones no contempladas con `else`.

## Iteración (for/while)

Los bucles nos permiten iterar durante un número definido de veces

```julia
for i in 1:3
    # Hacer algo
end
```

o bien, hasta que se cumpla una condición concreta. En este caso debemos indicar que `i` se trata de una variable `global` de forma que es accesible desde fuera del flujo (es decir, no `local` al bucle en curso).

```julia
i = 0

while i < 3
    # Hacer algo

    # Esto es necesario para que salga del bucle
    global i += 1
end
```

También podemos condicionar el bucle parando `break` o continuando a la siguiente iteración `continue`.

```julia
i = 1;

while true
    println(i)
    if i >= 3
        break
    end
    global i += 1
end
```

```julia
for i = 1:10
    if i % 3 != 0
        continue
    end
    println(i)
end
```

Más detalle en [Control flows](https://docs.julialang.org/en/v1/manual/control-flow/).