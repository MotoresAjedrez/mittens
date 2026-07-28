#!/bin/bash
# Validacion reproducible del proyecto. Usa --full para ejecutar tambien la
# suite perft completa y los diagnosticos mas costosos.
set -euo pipefail

cd "$(dirname "$0")"

cargo fmt --check
cargo check
cargo test
cargo build --release

if [[ "${1:-}" == "--full" ]]; then
  cargo run --release -- perft
  cargo run --release -- matetest
  cargo run --release -- seetest
  cargo run --release -- repetitiontest
fi

echo "Validacion terminada correctamente."
