# MiMotor Tal — pinlegal_v2 (fuente limpia)

Esta copia está preparada para experimentar con Razoring y SEE Pruning sin
mezclar HalfKP ni Correction History.

Incluye:

- evaluación clásica incremental (`EvalState`);
- atajo de legalidad por piezas clavadas;
- magic bitboards;
- NNUE plana de 770 entradas;
- salida UCI `info depth` también con Threads/Lazy SMP;
- arreglo `nnue_solicitada`: `UseNNUE=true` funciona aunque llegue antes de
  `NNUEPath`.

La producción reportada por Tavito es SHA-256
`205a580cd480eda70c0090e76f10aff36ca79fffcb5d388e546ebf4a7ed349b8`.
La fuente se reconstruyó aislando los componentes pinlegal sobre la versión
con evaluación incremental disponible localmente; el binario local se debe
validar con `cargo build --release` antes de compararlo.

Validación local de esta copia:

```bash
cargo test --locked
cargo build --release --locked
```

La red plana está en `pesos_v1.bin`. No incluye los cambios experimentales de
HalfKP ni Correction History.
