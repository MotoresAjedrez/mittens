// Claves Zobrist para el hash incremental del tablero.
// Generadas con un PRNG determinista (splitmix64) semillado con una
// constante fija -- no hace falta que sean "aleatorias de verdad", solo
// que no tengan patrones que generen colisiones sistemáticas.

pub struct ZobristKeys {
    pub piece_square: [[[u64; 64]; 6]; 2], // [color][piece_type][square]
    pub castling: [u64; 16],               // indexado por el byte de derechos (4 bits)
    pub en_passant_file: [u64; 8],
    pub side_to_move: u64,
}

struct SplitMix64(u64);

impl SplitMix64 {
    const fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// MISMO generador, MISMA semilla y MISMO orden de consumo que la version
/// anterior (que construia la tabla en tiempo de ejecucion dentro de un
/// OnceLock): los 793 numeros salen identicos, y el test
/// `const_coincide_con_construccion_en_ejecucion` lo comprueba bit a bit.
/// La diferencia es que ahora se calculan al COMPILAR: `keys()` pasa de ser
/// una consulta a OnceLock (carga atomica con orden Acquire + salto) a
/// devolver la direccion de un static ya inicializado. Como `keys()` se llama
/// dos veces por cada pieza movida dentro de make_move (add_piece /
/// remove_piece), esa carga atomica estaba en el camino mas caliente del
/// motor.
const fn build_keys() -> ZobristKeys {
    let mut rng = SplitMix64(0x5EED_C0FF_EE15_BAAD);
    let mut piece_square = [[[0u64; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut s = 0;
            while s < 64 {
                piece_square[c][p][s] = rng.next();
                s += 1;
            }
            p += 1;
        }
        c += 1;
    }
    let mut castling = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        castling[i] = rng.next();
        i += 1;
    }
    let mut en_passant_file = [0u64; 8];
    let mut i = 0;
    while i < 8 {
        en_passant_file[i] = rng.next();
        i += 1;
    }
    let side_to_move = rng.next();

    ZobristKeys {
        piece_square,
        castling,
        en_passant_file,
        side_to_move,
    }
}

static KEYS: ZobristKeys = build_keys();

#[inline(always)]
pub fn keys() -> &'static ZobristKeys {
    &KEYS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// La tabla constante debe ser identica, numero a numero, a la que
    /// producia el constructor en tiempo de ejecucion.
    #[test]
    fn const_coincide_con_construccion_en_ejecucion() {
        let mut rng = SplitMix64(0x5EED_C0FF_EE15_BAAD);
        let k = keys();
        for c in 0..2 {
            for p in 0..6 {
                for s in 0..64 {
                    assert_eq!(k.piece_square[c][p][s], rng.next());
                }
            }
        }
        for i in 0..16 {
            assert_eq!(k.castling[i], rng.next());
        }
        for i in 0..8 {
            assert_eq!(k.en_passant_file[i], rng.next());
        }
        assert_eq!(k.side_to_move, rng.next());
    }
}
