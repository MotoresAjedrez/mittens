# Acumulador clásico incremental — candidato aislado

Base: `mi-motor-rust-magic` / producción Magic + Hindsight, SHA
`b45933e252009b73c0af1a032127ebc690712c8bbfda23b13afbbddf96c870ed`.

## Cambio

- Se añadió `ClassicalAccumulator` como estado lateral de búsqueda; `Board`
  no cambia de tamaño ni de semántica.
- Material, PST, fase y cantidad de piezas no-peón se actualizan por delta de
  bitboards.
- Estructura de peones, peones pasados, escudo de rey y actividad de torres se
  recalculan solo cuando cambia un peón, rey o torre.
- Movilidad y unidades/atacantes de la zona del rey se actualizan de forma
  exacta: peones/caballos por delta de pieza, y alfiles/torres/damas a partir
  de todos los rayos afectados antes/después de cada cambio de ocupación.
- `evaluate_with_nnue` lento se conserva como oráculo; la búsqueda usa
  `evaluate_with_state` con el sidecar clásico + acumulador NNUE.

## Correctitud verificada

- `cargo fmt --check`.
- `cargo test --locked`: 24/24.
- Perft completo: posiciones inicial, Kiwipete, posición 3 y posición 5 en
  las profundidades incluidas, todos exactos.
- `matetest`, `seetest` (incluye 11,275 capturas frente al oráculo),
  `repetitiontest` y `endgametest`: todos pasan.
- Pruebas diferenciales del acumulador contra la evaluación lenta para Tal y
  Universal, movimientos normales, capturas, doble avance, ambos en passant,
  los cuatro enroques, promociones/captura-promociones, rayos de sliders,
  null move y fuzz determinista.

## Rendimiento reproducible (Threads=1, Hash=128, NNUE plana de producción)

Posición fija: `r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6`.

| Binario | Nodos | Score / mejor jugada | NPS |
|---|---:|---|---:|
| Producción Magic, corrida 1 | 6,204,580 | +146, c3d5 | 515,288 |
| Producción Magic, corrida 2 | 6,204,580 | +146, c3d5 | 516,919 |
| Candidato incremental, corrida 1 | 6,204,580 | +146, c3d5 | 736,974 |
| Candidato incremental, corrida 2 | 6,204,580 | +146, c3d5 | 729,436 |

La mediana sube aproximadamente de 516k a 733k NPS (+42%). Los nodos,
score, profundidad 12 y mejor jugada coinciden en la posición medida.

## Perfil

Se guardaron perfiles `sample` antes/después en `measurements/`. El nuevo
perfil ya no muestra el escaneo clásico completo por nodo dentro de la función
de evaluación; el coste se desplaza a `EvalState::despues_de_jugada`, que
actualiza solo las piezas/rayos afectados. La medición final de velocidad se
hizo sin profiler, porque los símbolos de depuración reducen NPS.

## H2H final

H2H aislado contra producción Magic/Hindsight, 40 partidas a 600 ms/jugada,
con los mismos pesos y aperturas pareadas:

```text
incremental_classical_40: +14 =16 -10, 55.0%, AMBIGUA
```

El resultado quedó exactamente en 55.0%, por lo que no superaba el umbral
habitual de `>55%`. Posteriormente el usuario autorizó explícitamente el
despliegue pese a la ambigüedad; se instaló como excepción documentada.

Despliegue: SHA `151e9b4af65aea070168e330eacfcf0bc9e6f5ca808de8ee7513b4bea87fb755`.
Respaldo previo: `mimotor-tal-rust.pre-incremental-20260713-164617`.
