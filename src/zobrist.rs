// Claves Zobrist para el hash incremental del tablero.
// Generadas con un PRNG determinista (splitmix64) semillado con una
// constante fija -- no hace falta que sean "aleatorias de verdad", solo
// que no tengan patrones que generen colisiones sistemáticas.

use std::sync::OnceLock;

pub struct ZobristKeys {
    pub piece_square: [[[u64; 64]; 6]; 2], // [color][piece_type][square]
    pub castling: [u64; 16],               // indexado por el byte de derechos (4 bits)
    pub en_passant_file: [u64; 8],
    pub side_to_move: u64,
}

static KEYS: OnceLock<ZobristKeys> = OnceLock::new();

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn build_keys() -> ZobristKeys {
    let mut rng = SplitMix64(0x5EED_C0FF_EE15_BAAD);
    let mut piece_square = [[[0u64; 64]; 6]; 2];
    for color in &mut piece_square {
        for pieza in color {
            for casilla in pieza {
                *casilla = rng.next();
            }
        }
    }
    let mut castling = [0u64; 16];
    for derecho in &mut castling {
        *derecho = rng.next();
    }
    let mut en_passant_file = [0u64; 8];
    for archivo in &mut en_passant_file {
        *archivo = rng.next();
    }
    let side_to_move = rng.next();

    ZobristKeys {
        piece_square,
        castling,
        en_passant_file,
        side_to_move,
    }
}

pub fn keys() -> &'static ZobristKeys {
    KEYS.get_or_init(build_keys)
}
