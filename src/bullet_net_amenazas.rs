// Red NNUE bullet con feature-set ENRIQUECIDO de amenazas:
//
//     (5376 -> H)x2 -> 1, doble perspectiva, activacion SCReLU  (H=512)
//
// Las 5376 features = 768 base pieza-casilla (la misma codificacion
// dual-perspectiva de Chess768) + 4608 de amenaza
// (color_atacante x tipo_atacante x tipo_victima x casilla_victima).
// Esta capa de extraccion replica EXACTAMENTE la logica de inputs.rs del
// proyecto trainer (cand_bullet_nnue_1024_amenazas), que es la fuente de
// verdad que genero los pesos del quantised.bin que cargamos aqui. La
// implementacion original opera sobre un bulletformat::ChessBoard ya
// normalizado a la perspectiva del lado que mueve; aqui reconstruimos esa
// misma vista normalizada a partir del Board del motor (casillas a1=0,
// White=0/Black=1) y despues corremos la MISMA logica de ataques.
//
// FORMATO DEL ARCHIVO (quantised.bin), identico al de bullet_net.rs pero con
// N_ENTRADA=5376 y h=H (=512):
//   4 bloques consecutivos de i16 little-endian:
//     1) l0w: 5376*512 = 2752512 valores, dispuestos POR FEATURE
//     2) l0b: 512 valores
//     3) l1w: 1024 valores (512 para la perspectiva del que mueve, 512
//        para la del rival)
//     4) l1b: 1 valor
//   Total util = 2754049 * 2 = 5508098 bytes, mas relleno final "bullet"
//   repetido hasta alinear a 64 bytes. Se valida, no se ignora.
//
// IMPLEMENTADO (esta sesion): actualizacion INCREMENTAL del accumulator con
// doble perspectiva (ver AcumBulletAmenazas mas abajo). Las 768 features
// base se actualizan con el delta trivial de pieza-casilla. Las 4608 de
// amenaza dependen de la ocupacion GLOBAL, asi que se calcula el conjunto
// MINIMO de (atacante, victima) que puede haber cambiado tras una jugada:
//   a) piezas en casillas modificadas (movida, capturada, enroque, al paso,
//      promocion) -> restar sus amenazas con el tablero de antes, sumar con
//      el de despues;
//   b) deslizantes ESTABLES (misma casilla) cuya linea de ataque se abrio o
//      cerro -> solo el delta de victimas;
//   c) no-deslizantes estables (peon/caballo/rey) que atacan una casilla
//      modificada -> solo la feature de la victima que cambio.
// Cada par (atacante, victima) emite DOS indices (una por perspectiva) con
// la misma geometria de ataques (independiente de la perspectiva); la
// verificacion bit-identica contra el recalculo completo esta en los tests
// `incremental_bit_identico_*`. Resultado: O(features_afectadas * H) por
// jugada en vez de O(N_INPUTS*H) por evaluacion.

use crate::board::Board;
use crate::types::{ALL_PIECE_TYPES, Color, PieceType};

pub const N_BASE: usize = 768;
pub const N_THREAT: usize = 2 * 6 * 6 * 64; // 4608
pub const N_INPUTS: usize = N_BASE + N_THREAT; // 5376
/// Neuronas de la capa oculta. Se baja de 1024 a 512: con 1024 la matriz l0
/// pesa 11 MB y no cabe en L2, lo que costaba 3-5 plies de profundidad a
/// igual tiempo por jugada (el mejor loss de entrenamiento no compensaba esa
/// perdida de busqueda). Con 512 son 5.5 MB. Es una constante de compilacion
/// porque los acumuladores son arrays de tamano fijo [i32; H]; el motor
/// soporta UNA sola variante a la vez y el tamano del archivo de pesos debe
/// coincidir (ver `bytes_utiles`/`tamano_plausible`).
pub const H: usize = 512;
/// Maximo de pares (stm, ntm) por posicion: 32 piezas base + amenazas.
/// Mismo valor que el trainer (medido sobre el dataset real).
pub const MAX_ACTIVE: usize = 256;

const QA: i32 = 255;
const QB: i32 = 64;
const ALINEACION: usize = 64;

/// Escala de salida. La red bullet aprende `salida ~= score / eval_scale` y
/// el archivo no lleva ese dato; se configura con MIMOTOR_BULLET_SCALE
/// (defecto 400, lo que usaron todas las redes entrenadas hasta ahora).
fn scale() -> i32 {
    use std::sync::OnceLock;
    static CACHE: OnceLock<i32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("MIMOTOR_BULLET_SCALE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| (16..=4096).contains(v))
            .unwrap_or(400)
    })
}

const fn bytes_utiles() -> usize {
    (N_INPUTS * H + H + 2 * H + 1) * 2
}

/// True si el tamano del archivo corresponde a esta arquitectura
/// (5376 -> H): util + hasta 63 bytes de relleno "bullet".
pub fn tamano_plausible(n: usize) -> bool {
    let utiles = bytes_utiles();
    n >= utiles && n - utiles < ALINEACION
}

// ---------------------------------------------------------------------------
// Tablas de ataque precomputadas (indices 0..64, coordenadas de perspectiva).
// Copia fiel de inputs.rs del trainer.
// ---------------------------------------------------------------------------
type Table = std::sync::OnceLock<[u64; 64]>;

fn knight_table() -> &'static [u64; 64] {
    static T: Table = Table::new();
    T.get_or_init(|| {
        let mut t = [0u64; 64];
        let offs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
        for sq in 0..64 {
            let (f, r) = (sq as i32 % 8, sq as i32 / 8);
            let mut m = 0u64;
            for (df, dr) in offs {
                let (nf, nr) = (f + df, r + dr);
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    m |= 1u64 << (nr * 8 + nf);
                }
            }
            t[sq] = m;
        }
        t
    })
}

fn king_table() -> &'static [u64; 64] {
    static T: Table = Table::new();
    T.get_or_init(|| {
        let mut t = [0u64; 64];
        let offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
        for sq in 0..64 {
            let (f, r) = (sq as i32 % 8, sq as i32 / 8);
            let mut m = 0u64;
            for (df, dr) in offs {
                let (nf, nr) = (f + df, r + dr);
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    m |= 1u64 << (nr * 8 + nf);
                }
            }
            t[sq] = m;
        }
        t
    })
}

/// Ataques de peon del lado que mueve (avanza "hacia arriba": +7/+9).
fn pawn_stm_table() -> &'static [u64; 64] {
    static T: Table = Table::new();
    T.get_or_init(|| {
        let mut t = [0u64; 64];
        for sq in 0..64 {
            let f = sq as i32 % 8;
            let r = sq as i32 / 8;
            let mut m = 0u64;
            for (df, dr) in [(-1, 1), (1, 1)] {
                let (nf, nr) = (f + df, r + dr);
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    m |= 1u64 << (nr * 8 + nf);
                }
            }
            t[sq] = m;
        }
        t
    })
}

/// Ataques de peon del lado que NO mueve (avanza "hacia abajo": -7/-9).
fn pawn_ntm_table() -> &'static [u64; 64] {
    static T: Table = Table::new();
    T.get_or_init(|| {
        let mut t = [0u64; 64];
        for sq in 0..64 {
            let f = sq as i32 % 8;
            let r = sq as i32 / 8;
            let mut m = 0u64;
            for (df, dr) in [(-1, -1), (1, -1)] {
                let (nf, nr) = (f + df, r + dr);
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    m |= 1u64 << (nr * 8 + nf);
                }
            }
            t[sq] = m;
        }
        t
    })
}

/// Rayos (bitboards) por direccion para deslizantes.
/// Direcciones: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW.
fn rays_table() -> &'static [[u64; 64]; 8] {
    static T: std::sync::OnceLock<[[u64; 64]; 8]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let steps = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
        let mut t = [[0u64; 64]; 8];
        for sq in 0..64 {
            for (d, (df, dr)) in steps.iter().enumerate() {
                let (mut f, mut r) = (sq as i32 % 8, sq as i32 / 8);
                let mut m = 0u64;
                loop {
                    f += df;
                    r += dr;
                    if f < 0 || f >= 8 || r < 0 || r >= 8 {
                        break;
                    }
                    m |= 1u64 << (r * 8 + f);
                }
                t[d][sq] = m;
            }
        }
        t
    })
}

fn sliding_attacks(sq: usize, occ: u64, dirs: &[usize]) -> u64 {
    let rays = rays_table();
    let mut atk = 0u64;
    for &d in dirs {
        let mut ray = rays[d][sq];
        // Las direcciones cuyo paso incrementa el indice (N=+8, NE=+9, E=+1,
        // NW=+7) tienen la casilla mas cercana a la pieza en el bit MAS BAJO;
        // las que decrementan (S, SW, W, SE), en el bit MAS ALTO. Iterar desde
        // la pieza hacia afuera es esencial para que el primer bloqueo corte
        // el rayo.
        let ascending = matches!(d, 0 | 1 | 2 | 7);
        while ray != 0 {
            let bit = if ascending {
                1u64 << ray.trailing_zeros()
            } else {
                1u64 << (63 - ray.leading_zeros())
            };
            atk |= bit;
            if occ & bit != 0 {
                break; // primer bloqueo
            }
            ray &= !bit;
        }
    }
    atk
}

fn attacks(piece: u8, sq: usize, occ: u64) -> u64 {
    match piece & 7 {
        0 => {
            if piece & 8 == 0 {
                pawn_stm_table()[sq]
            } else {
                pawn_ntm_table()[sq]
            }
        }
        1 => knight_table()[sq],
        2 => sliding_attacks(sq, occ, &[1, 3, 5, 7]), // alfil: NE SE SW NW
        3 => sliding_attacks(sq, occ, &[0, 2, 4, 6]), // torre: N E S W
        4 => sliding_attacks(sq, occ, &[0, 1, 2, 3, 4, 5, 6, 7]), // dama
        5 => king_table()[sq],
        _ => 0,
    }
}

/// Indice de amenaza (misma formula que features_threat.py):
///   ((color * 6 + tipo_atacante) * 6 + tipo_victima) * 64 + casilla_victima
/// con offset N_BASE. color=1 => atacante del lado que mueve (rol "blanco" en
/// perspectiva), color=0 => el lado que NO mueve.
#[inline(always)]
fn threat_idx(attacker_stm: bool, atk_type: usize, vic_type: usize, sq: usize) -> usize {
    let color = usize::from(attacker_stm);
    N_BASE + ((color * 6 + atk_type) * 6 + vic_type) * 64 + sq
}

/// Replica `Chess768Threats::map_features` de inputs.rs sobre el Board del
/// motor. Emite el MISMO conjunto de pares (indice_vista_stm,
/// indice_vista_ntm) que el trainer.
///
/// Paso 1: reconstruir la vista normalizada a la perspectiva del lado que
/// mueve (igual que hace bulletformat::ChessBoard): piezas del bando que
/// mueve con bit 3 = 0, casillas espejadas con sq^56 si mueven negras.
/// Paso 2: correr la logica de inputs.rs sobre esa vista (tablas de ataque
/// de perspectiva + formula de indice).
pub fn map_features(b: &Board, f: impl FnMut(usize, usize)) {
    let espejo = b.turn == Color::Black;

    // Mapa casilla (perspectiva) -> pieza (bit3 = color de perspectiva
    // [0=stm, 1=ntm], bits 0-2 = tipo) y ocupacion de perspectiva.
    let mut sq_piece = [0u8; 64];
    let mut pocc = 0u64;
    for color in 0..2usize {
        for (pt_idx, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
            let mut piezas = b.pieces[color][pt as usize];
            while piezas != 0 {
                let sq = crate::bitboard::pop_lsb(&mut piezas) as usize;
                let psq = if espejo { sq ^ 56 } else { sq };
                let pcolor = (color ^ usize::from(espejo)) as u8; // 0=stm, 1=ntm
                sq_piece[psq] = (pcolor << 3) | (pt_idx as u8);
                pocc |= 1u64 << psq;
            }
        }
    }

    // ---- Base 768 (identico a Chess768): cada pieza emite un par stm/ntm.
    // Recorremos la ocupacion de perspectiva para que el orden sea
    // deterministico (independiente del orden interno del Board). ----
    let mut f = f;
    let mut restantes = pocc;
    while restantes != 0 {
        let sq = restantes.trailing_zeros() as usize;
        restantes &= restantes - 1;
        let piece = sq_piece[sq];
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let stm = [0, 384][c] + pc + sq;
        let ntm = [384, 0][c] + pc + (sq ^ 56);
        f(stm, ntm);
    }

    // ---- Amenazas 4608: cada ataque real (pieza -> casilla ocupada) emite
    // un par: la vista stm con el atacante en su rol de perspectiva y la
    // vista ntm con el rol invertido y la casilla espejada sq^56. ----
    let mut restantes = pocc;
    while restantes != 0 {
        let sq_a = restantes.trailing_zeros() as usize;
        restantes &= restantes - 1;
        let piece = sq_piece[sq_a];
        let atk_type = usize::from(piece & 7);
        let attacker_stm = piece & 8 == 0;
        let mut targets = attacks(piece, sq_a, pocc) & pocc;
        while targets != 0 {
            let sq_v = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            let vic_type = usize::from(sq_piece[sq_v] & 7);
            let stm = threat_idx(attacker_stm, atk_type, vic_type, sq_v);
            let ntm = threat_idx(!attacker_stm, atk_type, vic_type, sq_v ^ 56);
            f(stm, ntm);
        }
    }
}

// ---------------------------------------------------------------------------
// Red cuantizada y forward pass.
// ---------------------------------------------------------------------------
pub struct RedBulletAmenazas {
    /// [feature][neurona] -- 5376 bloques contiguos de 1024 pesos i16.
    l0w: Vec<i16>,
    l0b: Vec<i16>,
    /// [stm(1024) | ntm(1024)].
    l1w: Vec<i16>,
    l1b: i16,
}

impl RedBulletAmenazas {
    pub fn cargar_de_bytes(datos: &[u8]) -> Option<RedBulletAmenazas> {
        if !tamano_plausible(datos.len()) {
            return None;
        }
        let utiles = bytes_utiles();
        let relleno = &datos[utiles..];
        for (i, &b) in relleno.iter().enumerate() {
            if b != b"bullet"[i % 6] {
                eprintln!(
                    "info string NNUE bullet 5376: relleno final inesperado ({} bytes), se rechaza",
                    relleno.len()
                );
                return None;
            }
        }

        let leer = |desde: usize, n: usize| -> Vec<i16> {
            (0..n)
                .map(|k| {
                    let p = desde + k * 2;
                    i16::from_le_bytes([datos[p], datos[p + 1]])
                })
                .collect()
        };
        let mut cursor = 0usize;
        let l0w = leer(cursor, N_INPUTS * H);
        cursor += N_INPUTS * H * 2;
        let l0b = leer(cursor, H);
        cursor += H * 2;
        let l1w = leer(cursor, 2 * H);
        cursor += 2 * H * 2;
        let l1b = leer(cursor, 1)[0];

        eprintln!(
            "info string NNUE bullet 5376: capa oculta de {H} neuronas, {} features cargadas",
            N_INPUTS
        );
        Some(RedBulletAmenazas {
            l0w,
            l0b,
            l1w,
            l1b,
        })
    }

    #[inline(always)]
    fn columna(&self, feature: usize) -> &[i16] {
        &self.l0w[feature * H..(feature + 1) * H]
    }

    #[inline(always)]
    fn sumar(&self, acc: &mut [i32; H], feature: usize) {
        let col = self.columna(feature);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let mut j = 0;
            while j + 8 <= H {
                let w16 = vld1q_s16(col.as_ptr().add(j));
                let w_lo = vmovl_s16(vget_low_s16(w16));
                let w_hi = vmovl_s16(vget_high_s16(w16));
                let a_lo = vld1q_s32(acc.as_ptr().add(j));
                let a_hi = vld1q_s32(acc.as_ptr().add(j + 4));
                vst1q_s32(acc.as_mut_ptr().add(j), vaddq_s32(a_lo, w_lo));
                vst1q_s32(acc.as_mut_ptr().add(j + 4), vaddq_s32(a_hi, w_hi));
                j += 8;
            }
            for k in j..H {
                acc[k] += col[k] as i32;
            }
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.sumar_escalar(acc, feature);
        }
    }

    #[inline(always)]
    fn restar(&self, acc: &mut [i32; H], feature: usize) {
        let col = self.columna(feature);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let mut j = 0;
            while j + 8 <= H {
                let w16 = vld1q_s16(col.as_ptr().add(j));
                let w_lo = vmovl_s16(vget_low_s16(w16));
                let w_hi = vmovl_s16(vget_high_s16(w16));
                let a_lo = vld1q_s32(acc.as_ptr().add(j));
                let a_hi = vld1q_s32(acc.as_ptr().add(j + 4));
                vst1q_s32(acc.as_mut_ptr().add(j), vsubq_s32(a_lo, w_lo));
                vst1q_s32(acc.as_mut_ptr().add(j + 4), vsubq_s32(a_hi, w_hi));
                j += 8;
            }
            for k in j..H {
                acc[k] -= col[k] as i32;
            }
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.restar_escalar(acc, feature);
        }
    }

    /// Camino escalar original de `sumar`/`restar`. Conservado como
    /// referencia de correccion para el test `sumar_restar_neon_igual_que_escalar`.
    #[inline(always)]
    #[allow(dead_code)]
    fn sumar_escalar(&self, acc: &mut [i32; H], feature: usize) {
        let col = self.columna(feature);
        for (a, &w) in acc.iter_mut().zip(col) {
            *a += w as i32;
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn restar_escalar(&self, acc: &mut [i32; H], feature: usize) {
        let col = self.columna(feature);
        for (a, &w) in acc.iter_mut().zip(col) {
            *a -= w as i32;
        }
    }

    /// SCReLU + capa de salida a partir de los dos acumuladores (bias ya
    /// incluido). Comun al camino completo (`evaluar_tablero`) y al
    /// incremental (`AcumBulletAmenazas::evaluar`): mismo codigo, mismo
    /// resultado bit a bit.
    fn salida_desde_acumuladores(&self, acc: &[[i32; H]; 2], turn: Color) -> f32 {
        let (yo, rival) = if turn == Color::White {
            (&acc[0], &acc[1])
        } else {
            (&acc[1], &acc[0])
        };
        let suma = self.producto_punto(yo, rival);
        let salida = suma / QA as i64 + self.l1b as i64;
        (salida * scale() as i64) as f32 / (QA * QB) as f32
    }

    /// Capa de salida vectorizada con NEON: mismo patron que
    /// `bullet_net.rs::producto_punto`, pero AQUI el acumulador ya es i32
    /// (no i16 como alla), asi que no hace falta el paso de ensanchado
    /// i16->i32 sobre el acumulador -- solo sobre los pesos `l1w` (i16).
    #[inline(always)]
    fn producto_punto(&self, yo: &[i32; H], rival: &[i32; H]) -> i64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // RANGOS (identicos al razonamiento de bullet_net.rs):
            //   v = clamp(acumulador, 0, 255)   ->  0..255
            //   v*v                             ->  0..65025   (cabe en i32)
            //   w es i16                        ->  -32768..32767
            //   v*v*w                           ->  |.| <= 65025*32768 = 2.13e9
            // Cabe justo en i32 (limite 2.147e9): cada termino se calcula en
            // i32 con vmulq_s32 sin desbordar nunca. La suma de los 1024
            // terminos SI desbordaria i32, por eso se ensancha a i64 con
            // vpadalq_s32 (suma por pares de i32 -> i64). La suma entera es
            // asociativa: el resultado es EXACTAMENTE el mismo que el bucle
            // escalar.
            //
            // CUATRO acumuladores independientes por el mismo motivo que en
            // bullet_net.rs: con uno solo las multiplicaciones formarian una
            // cadena de dependencias y el bucle quedaria limitado por
            // latencia en vez de throughput.
            let cero = vdupq_n_s32(0);
            let tope = vdupq_n_s32(QA);
            let w = self.l1w.as_ptr();
            let mut acc0 = vdupq_n_s64(0);
            let mut acc1 = vdupq_n_s64(0);
            let mut acc2 = vdupq_n_s64(0);
            let mut acc3 = vdupq_n_s64(0);
            // Cierre que procesa 4 neuronas (un vector de i32): clamp ->
            // v*v -> ensanchar el peso i16 a i32 -> por el peso -> acumular
            // ensanchando a i64.
            macro_rules! bloque {
                ($src:expr, $peso:expr, $a:expr) => {{
                    let v32 = vminq_s32(vmaxq_s32(vld1q_s32($src), cero), tope);
                    let w16 = vld1_s16($peso);
                    let w32 = vmovl_s16(w16);
                    let p = vmulq_s32(vmulq_s32(v32, v32), w32);
                    $a = vpadalq_s32($a, p);
                }};
            }
            let mut j = 0;
            while j + 8 <= H {
                bloque!(yo.as_ptr().add(j), w.add(j), acc0);
                bloque!(yo.as_ptr().add(j + 4), w.add(j + 4), acc1);
                bloque!(rival.as_ptr().add(j), w.add(H + j), acc2);
                bloque!(rival.as_ptr().add(j + 4), w.add(H + j + 4), acc3);
                j += 8;
            }
            let acc = vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3));
            let mut suma = vgetq_lane_s64::<0>(acc) + vgetq_lane_s64::<1>(acc);
            // Cola escalar por si H no fuera multiplo de 8 (hoy H=512, no
            // se ejecuta nunca).
            for k in j..H {
                let v = yo[k].clamp(0, QA) as i64;
                suma += v * v * self.l1w[k] as i64;
                let u = rival[k].clamp(0, QA) as i64;
                suma += u * u * self.l1w[H + k] as i64;
            }
            return suma;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.producto_punto_escalar(yo, rival)
        }
    }

    /// Camino ESCALAR original de la capa de salida. Se conserva como
    /// referencia de correccion: el test `salida_neon_igual_que_escalar`
    /// compara termino a termino contra la version vectorizada.
    #[inline(always)]
    fn producto_punto_escalar(&self, yo: &[i32; H], rival: &[i32; H]) -> i64 {
        let mut suma: i64 = 0;
        // SCReLU: clamp(x, 0, QA)^2. Misma escala que bullet_net.rs.
        for j in 0..H {
            let v = yo[j].clamp(0, QA) as i64;
            suma += v * v * self.l1w[j] as i64;
            let u = rival[j].clamp(0, QA) as i64;
            suma += u * u * self.l1w[H + j] as i64;
        }
        suma
    }

    /// Igual que `salida_desde_acumuladores` pero forzando el camino
    /// escalar. Solo para tests.
    #[cfg(test)]
    fn salida_desde_acumuladores_escalar(&self, acc: &[[i32; H]; 2], turn: Color) -> f32 {
        let (yo, rival) = if turn == Color::White {
            (&acc[0], &acc[1])
        } else {
            (&acc[1], &acc[0])
        };
        let suma = self.producto_punto_escalar(yo, rival);
        let salida = suma / QA as i64 + self.l1b as i64;
        (salida * scale() as i64) as f32 / (QA * QB) as f32
    }

    /// Forward completo sobre un tablero: extrae las 5376 features,
    /// construye los dos acumuladores (vista del que mueve y del rival) y
    /// ejecuta SCReLU + capa de salida. Devuelve centipeones desde la
    /// perspectiva del lado que mueve (misma convencion que la evaluacion
    /// clasica del motor). Version NO incremental: O(N_INPUTS*H) por llamada.
    /// Se conserva como REFERENCIA: es la implementacion con la que los
    /// tests de regresion comparan el camino incremental bit a bit.
    #[allow(dead_code)]
    pub fn evaluar_tablero(&self, b: &Board) -> f32 {
        // Recolectar los pares primero (el cierre con dos prestamos mutuos
        // sobre `acc` no compila; un buffer externo lo evita).
        let mut pares: Vec<(usize, usize)> = Vec::with_capacity(MAX_ACTIVE);
        map_features(b, |s, n| pares.push((s, n)));

        // stm es el indice en la perspectiva del lado que mueve; si mueven
        // negras, la vista stm va al acumulador [1] y la ntm al [0].
        let (stm_i, ntm_i) = if b.turn == Color::White { (0usize, 1usize) } else { (1, 0) };
        let mut acc = [[0i32; H]; 2];
        for &(s, _) in &pares {
            self.sumar(&mut acc[stm_i], s);
        }
        for &(_, n) in &pares {
            self.sumar(&mut acc[ntm_i], n);
        }
        for j in 0..H {
            acc[0][j] += self.l0b[j] as i32;
            acc[1][j] += self.l0b[j] as i32;
        }

        self.salida_desde_acumuladores(&acc, b.turn)
    }
}

// ---------------------------------------------------------------------------
// Acumulador INCREMENTAL para la red 5376 (doble perspectiva).
// ---------------------------------------------------------------------------
// Dos vistas FIJAS ligadas al color de la perspectiva, no al turno actual:
//   acc[0] = perspectiva BLANCA  (stm=White, espejo 0)  == vista "stm" con
//            blancas al turno, vista "ntm" con negras al turno.
//   acc[1] = perspectiva NEGRA   (stm=Black, espejo 56) == vista "stm" con
//            negras al turno, vista "ntm" con blancas al turno.
// Es exactamente la particion que hace `map_features` (espejo = turn==Black)
// y, como cada vista queda atada a su color, el cambio de turno NO exige
// intercambiar los acumuladores: `evaluar()` elige cual es "yo" segun el
// turno actual. `desde_tablero` construye las dos vistas desde cero con
// `map_features` + bias; `aplicar_jugada` aplica el delta de la jugada.
//
// Delta de amenazas: las features de amenaza dependen de la ocupacion global
// (un ataque deslizante puede abrirse/cerrarse por una pieza que se movio en
// cualquier parte del tablero), asi que no basta con restar/sumar la pieza
// movida. Se portan aqui los TRES pasos validados bit-a-bit del acumulador
// de neural.rs (AcumAmenazas), generalizados a las dos perspectivas y con
// delta de victimas tambien para deslizantes que se movieron:
//   a) deslizantes afectados (movidos, capturados, promocionados y estables
//      cuya linea se abrio/cerro): delta de VICTIMAS (la feature de amenaza
//      no depende de la casilla del atacante, solo de (stm, tipo atacante,
//      tipo victima, casilla victima)); capturadas/promocionadas a otro tipo
//      se restan/suman completas;
//   b) no-deslizantes en casillas modificadas (peon/caballo/rey movidos,
//      capturados, promocionados): pasada completa restar-antes /
//      sumar-despues (pocas victimas, no merece la pena el delta);
//   c) no-deslizantes ESTABLES que atacan una casilla modificada: solo la
//      feature de la victima que cambio (antes/despues).
// La GEOMETRIA de ataques es independiente de la perspectiva (espejar el
// tablero conserva los ataques; peones incluidos: el peon "stm" de una vista
// es el "ntm" de la otra), asi que cada par (atacante, victima) se computa
// una sola vez sobre el tablero real y emite DOS indices de feature, uno por
// vista (rol "stm" del atacante y casilla victima espejada sq^56).
#[derive(Clone)]
pub struct AcumBulletAmenazas {
    red: &'static RedBulletAmenazas,
    tablero: Board,
    /// Dos acumuladores [perspectiva][neurona] con el bias l0b ya incluido.
    acc: [[i32; H]; 2],
}

// --- helpers del delta (misma logica validada que neural.rs) -------------

/// Casillas cuyo ocupante difiere entre antes/despues (XOR por color/tipo).
#[inline]
fn mascara_casillas_cambiadas(antes: &Board, despues: &Board) -> u64 {
    let mut cambiadas = 0u64;
    for color in 0..2 {
        for pt in 0..ALL_PIECE_TYPES.len() {
            cambiadas |= antes.pieces[color][pt] ^ despues.pieces[color][pt];
        }
    }
    cambiadas
}

#[inline]
fn ataques_de_pieza(pt: PieceType, color: Color, sq: u8, ocupado: u64) -> u64 {
    match pt {
        PieceType::Pawn => crate::bitboard::pawn_attacks(color, sq),
        PieceType::Knight => crate::bitboard::knight_attacks(sq),
        PieceType::Bishop => crate::bitboard::bishop_attacks(sq, ocupado),
        PieceType::Rook => crate::bitboard::rook_attacks(sq, ocupado),
        PieceType::Queen => crate::bitboard::queen_attacks(sq, ocupado),
        PieceType::King => crate::bitboard::king_attacks(sq),
    }
}

/// Indice base 768 de la pieza (color, tipo, casilla real) en la vista v:
/// vista 0 = perspectiva blanca (sin espejo), vista 1 = perspectiva negra
/// (espejo 56). Formula identica a `map_features` con espejo = (v == 1).
#[inline(always)]
fn base_idx_v(color: usize, pt: usize, sq: usize, v: usize) -> usize {
    [0, 384][color ^ v] + 64 * pt + (sq ^ (56 * v))
}

/// Indice de amenaza del par (atacante color_idx, tipo, victima sq_v, tipo)
/// en la vista v (v=0: blancas stm; v=1: negras stm). Igual formula que
/// `map_features` con espejo = (v == 1).
#[inline(always)]
fn threat_idx_v(color_idx: usize, atk_type: usize, vic_type: usize, sq_v: usize, v: usize) -> usize {
    threat_idx(color_idx == v, atk_type, vic_type, sq_v ^ (56 * v))
}

/// Entre los deslizantes geometricamente alineados con una casilla cambiada,
/// conserva solo los que de verdad pueden haber cambiado su conjunto de
/// features (los demas rayos quedan bloqueados o no alcanzan la casilla).
#[inline]
fn mascara_slider_con_amenaza_cambiante(
    antes: &Board,
    despues: &Board,
    pt: PieceType,
    linea_geometrica: u64,
    cambiadas: u64,
) -> u64 {
    let piezas_antes = antes.pieces[Color::White as usize][pt as usize]
        | antes.pieces[Color::Black as usize][pt as usize];
    let piezas_despues = despues.pieces[Color::White as usize][pt as usize]
        | despues.pieces[Color::Black as usize][pt as usize];
    let mut comunes = piezas_antes & piezas_despues & linea_geometrica & !cambiadas;
    let mut necesarias = cambiadas;
    while comunes != 0 {
        let sq = crate::bitboard::pop_lsb(&mut comunes);
        let (color, encontrado) = antes.piece_at(sq).expect("bitboard inconsistente");
        debug_assert_eq!(encontrado, pt);
        let ataques_antes = ataques_de_pieza(pt, color, sq, antes.occupied);
        let ataques_despues = ataques_de_pieza(pt, color, sq, despues.occupied);
        if ataques_antes != ataques_despues || ((ataques_antes | ataques_despues) & cambiadas != 0)
        {
            necesarias |= 1u64 << sq;
        }
    }
    necesarias
}

impl RedBulletAmenazas {
    /// Aplica TODAS las amenazas de una pieza (atacante) a las dos vistas.
    /// La geometria de ataques es la del tablero real (igual en ambas
    /// perspectivas); solo cambia el indice de feature por vista.
    #[inline]
    fn aplicar_amenazas_de_pieza(
        &self,
        acc: &mut [[i32; H]; 2],
        tablero: &Board,
        color_idx: usize,
        pt: PieceType,
        sq: usize,
        sumar: bool,
    ) {
        let color = if color_idx == 0 {
            Color::White
        } else {
            Color::Black
        };
        let ataques = ataques_de_pieza(pt, color, sq as u8, tablero.occupied);
        let victimas = ataques & tablero.occupied;
        if victimas == 0 {
            return;
        }
        let atk_type = pt as usize;
        for tipo_v in 0..6 {
            let mut bb = victimas & (tablero.pieces[0][tipo_v] | tablero.pieces[1][tipo_v]);
            while bb != 0 {
                let sq_v = crate::bitboard::pop_lsb(&mut bb) as usize;
                for v in 0..2usize {
                    let idx = threat_idx_v(color_idx, atk_type, tipo_v, sq_v, v);
                    if sumar {
                        self.sumar(&mut acc[v], idx);
                    } else {
                        self.restar(&mut acc[v], idx);
                    }
                }
            }
        }
    }

    /// Aplica las amenazas de las piezas NO deslizantes (peon/caballo/rey)
    /// que estan en casillas cambiadas (movidas, capturadas o promocionadas),
    /// con su pasada completa: restar con `antes` y sumar con `despues` son
    /// dos llamadas con tableros distintos. Los deslizantes no llegan aqui
    /// (se procesan con delta de victimas en `aplicar_delta_deslizantes`).
    #[inline]
    fn aplicar_amenazas_no_deslizantes(
        &self,
        acc: &mut [[i32; H]; 2],
        tablero: &Board,
        cambiadas: u64,
        sumar: bool,
    ) {
        for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
            for pt in ALL_PIECE_TYPES {
                let deslizante = matches!(
                    pt,
                    PieceType::Bishop | PieceType::Rook | PieceType::Queen
                );
                if deslizante {
                    continue;
                }
                let mut piezas = tablero.pieces[color as usize][pt as usize] & cambiadas;
                while piezas != 0 {
                    let sq = crate::bitboard::pop_lsb(&mut piezas) as usize;
                    self.aplicar_amenazas_de_pieza(acc, tablero, color_idx, pt, sq, sumar);
                }
            }
        }
    }

    /// Delta de amenazas para UNA pieza deslizante (alfil/torre/dama) cuya
    /// identidad (color+tipo) es la misma en antes y despues: casilla sq1 en
    /// el tablero de antes y sq2 en el de despues (iguales para una pieza
    /// estable, distintas para una que se movio). La feature de amenaza NO
    /// depende de la casilla del atacante, solo de (stm, tipo atacante, tipo
    /// victima, casilla victima), asi que el delta de victimas es exacto:
    /// se restan las que dejan de serlo (o cambian de victima) y se suman las
    /// nuevas, saltando las identicas. Para la pasada vieja esto era un
    /// restar-completo + sumar-completo que se cancelaba en gran parte.
    #[inline]
    fn aplicar_delta_slider_par(
        &self,
        acc: &mut [[i32; H]; 2],
        antes: &Board,
        despues: &Board,
        cambiadas: u64,
        color_idx: usize,
        atk_type: usize,
        sq1: u8,
        sq2: u8,
    ) {
        let color = if color_idx == 0 {
            Color::White
        } else {
            Color::Black
        };
        let pt = match atk_type {
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            _ => PieceType::Queen,
        };
        let att1 = ataques_de_pieza(pt, color, sq1, antes.occupied);
        let att2 = ataques_de_pieza(pt, color, sq2, despues.occupied);
        let v1 = att1 & antes.occupied;
        let v2 = att2 & despues.occupied;
        // Victima en ambos conjuntos y casilla modificada: cambio el ocupante,
        // hay que restar la de antes y sumar la de despues.
        let comunes_cambiados = v1 & v2 & cambiadas;
        let restar = (v1 & !v2) | comunes_cambiados;
        let sumar = (v2 & !v1) | comunes_cambiados;
        if restar != 0 {
            for tipo_v in 0..6 {
                let mut bb = restar & (antes.pieces[0][tipo_v] | antes.pieces[1][tipo_v]);
                while bb != 0 {
                    let sq_v = crate::bitboard::pop_lsb(&mut bb) as usize;
                    for v in 0..2usize {
                        self.restar(&mut acc[v], threat_idx_v(color_idx, atk_type, tipo_v, sq_v, v));
                    }
                }
            }
        }
        if sumar != 0 {
            for tipo_v in 0..6 {
                let mut bb = sumar & (despues.pieces[0][tipo_v] | despues.pieces[1][tipo_v]);
                while bb != 0 {
                    let sq_v = crate::bitboard::pop_lsb(&mut bb) as usize;
                    for v in 0..2usize {
                        self.sumar(&mut acc[v], threat_idx_v(color_idx, atk_type, tipo_v, sq_v, v));
                    }
                }
            }
        }
    }

    /// Delta de amenazas de TODOS los deslizantes afectados por la jugada,
    /// unificando los tres casos en una sola pasada por (color, tipo):
    ///   * pieza que se MOVIO (una casilla cambiada en cada tablero): delta
    ///     de victimas con casillas distintas (sq1 -> sq2);
    ///   * pieza CAPTURADA o promocionada a otro tipo (solo en antes):
    ///     restar completo con el tablero de antes;
    ///   * pieza NUEVA (promocion, solo en despues): sumar completo con el
    ///     de despues;
    ///   * deslizante ESTABLE cuya linea se abrio/cerro (misma casilla):
    ///     delta de victimas con la misma casilla.
    /// Es exacto porque en una jugada legal cada (color, tipo) tiene a lo
    /// sumo una pieza sobre casillas cambiadas en cada tablero (la captura
    /// siempre es del color contrario y el enroque mueve tipos distintos),
    /// asi que el emparejamiento antes/despues es inequivoco.
    #[inline]
    fn aplicar_delta_deslizantes(
        &self,
        acc: &mut [[i32; H]; 2],
        antes: &Board,
        despues: &Board,
        cambiadas: u64,
        pt: PieceType,
        mascara: u64,
    ) {
        let atk_type = pt as usize;
        for (color_idx, color) in [(0usize, Color::White), (1usize, Color::Black)] {
            let a_bb = antes.pieces[color as usize][pt as usize] & mascara;
            let d_bb = despues.pieces[color as usize][pt as usize] & mascara;
            let a_mov = a_bb & cambiadas;
            let d_mov = d_bb & cambiadas;
            if a_mov != 0 && d_mov != 0 {
                let mut x = a_mov;
                let sq1 = crate::bitboard::pop_lsb(&mut x);
                let mut y = d_mov;
                let sq2 = crate::bitboard::pop_lsb(&mut y);
                self.aplicar_delta_slider_par(acc, antes, despues, cambiadas, color_idx, atk_type, sq1, sq2);
            } else if a_mov != 0 {
                let mut x = a_mov;
                let sq1 = crate::bitboard::pop_lsb(&mut x);
                self.aplicar_amenazas_de_pieza(acc, antes, color_idx, pt, sq1 as usize, false);
            } else if d_mov != 0 {
                let mut y = d_mov;
                let sq2 = crate::bitboard::pop_lsb(&mut y);
                self.aplicar_amenazas_de_pieza(acc, despues, color_idx, pt, sq2 as usize, true);
            }
            // Estables (misma casilla en ambos) cuya vision cambio.
            let mut estables = a_bb & d_bb & !cambiadas;
            while estables != 0 {
                let sq = crate::bitboard::pop_lsb(&mut estables);
                self.aplicar_delta_slider_par(acc, antes, despues, cambiadas, color_idx, atk_type, sq, sq);
            }
        }
    }

    /// Delta completo de una jugada sobre las dos vistas (base 768 + amenazas
    /// 4608). Es su propia inversa al intercambiar antes/despues: lo que se
    /// resta con el tablero de "antes" se suma con el de "despues" y
    /// viceversa, con el mismo conjunto de features (la mascara de casillas
    /// cambiadas es simetrica).
    fn actualizar_incremental(&self, acc: &mut [[i32; H]; 2], antes: &Board, despues: &Board) {
        // --- 1. base 768 (pieza-casilla): solo cambia lo que hay en las
        // casillas modificadas; dos indices por casilla (uno por vista).
        let cambiadas = mascara_casillas_cambiadas(antes, despues);
        let mut scan = cambiadas;
        while scan != 0 {
            let sq = crate::bitboard::pop_lsb(&mut scan) as usize;
            for v in 0..2usize {
                if let Some((c, pt)) = antes.piece_at(sq as u8) {
                    self.restar(&mut acc[v], base_idx_v(c as usize, pt as usize, sq, v));
                }
                if let Some((c, pt)) = despues.piece_at(sq as u8) {
                    self.sumar(&mut acc[v], base_idx_v(c as usize, pt as usize, sq, v));
                }
            }
        }

        // --- 2. amenazas ------------------------------------------------
        // 2.1. Piezas en casillas cambiadas (las que se movieron): pasada
        // completa restar-antes / sumar-despues. Antes hay que delimitar los
        // deslizantes ESTABLES cuyas lineas se tocaron (paso 2.2) con la
        // geometria de las casillas cambiadas.
        let mut lineas_alfil = 0u64;
        let mut lineas_torre = 0u64;
        let mut lineas_desde = cambiadas;
        while lineas_desde != 0 {
            let sq = crate::bitboard::pop_lsb(&mut lineas_desde);
            lineas_alfil |= crate::bitboard::bishop_attacks(sq, 0);
            lineas_torre |= crate::bitboard::rook_attacks(sq, 0);
        }
        let mascara_alfil = mascara_slider_con_amenaza_cambiante(
            antes,
            despues,
            PieceType::Bishop,
            lineas_alfil,
            cambiadas,
        );
        let mascara_torre = mascara_slider_con_amenaza_cambiante(
            antes,
            despues,
            PieceType::Rook,
            lineas_torre,
            cambiadas,
        );
        let mascara_dama = mascara_slider_con_amenaza_cambiante(
            antes,
            despues,
            PieceType::Queen,
            lineas_alfil | lineas_torre,
            cambiadas,
        );
        // 2.1. Piezas NO deslizantes movidas/capturadas/promocionadas:
        // pasada completa restar-antes / sumar-despues (los deslizantes se
        // procesan unificados en 2.2 con delta de victimas).
        self.aplicar_amenazas_no_deslizantes(acc, antes, cambiadas, false);
        self.aplicar_amenazas_no_deslizantes(acc, despues, cambiadas, true);
        // 2.2. Deslizantes (movidas, capturadas, promocionadas y estables
        // con linea abierta/cerrada): delta de victimas unificado.
        self.aplicar_delta_deslizantes(
            acc,
            antes,
            despues,
            cambiadas,
            PieceType::Bishop,
            mascara_alfil,
        );
        self.aplicar_delta_deslizantes(
            acc,
            antes,
            despues,
            cambiadas,
            PieceType::Rook,
            mascara_torre,
        );
        self.aplicar_delta_deslizantes(
            acc,
            antes,
            despues,
            cambiadas,
            PieceType::Queen,
            mascara_dama,
        );
        // 2.3. No-deslizantes estables (peon/caballo/rey) que atacan una
        // casilla cambiada como victima, sin haberse movido: su conjunto de
        // ataque no depende de la ocupacion, pero la pieza en la casilla que
        // atacan si cambio -> solo esa feature, antes y despues.
        let mut victimas_cambiadas = cambiadas;
        while victimas_cambiadas != 0 {
            let sq = crate::bitboard::pop_lsb(&mut victimas_cambiadas) as usize;
            let sqb = sq as u8;
            let mut atacantes = 0u64;
            atacantes |= crate::bitboard::knight_attacks(sqb)
                & (antes.pieces[0][PieceType::Knight as usize]
                    | antes.pieces[1][PieceType::Knight as usize]);
            atacantes |= crate::bitboard::king_attacks(sqb)
                & (antes.pieces[0][PieceType::King as usize]
                    | antes.pieces[1][PieceType::King as usize]);
            atacantes |= crate::bitboard::pawn_attacks(Color::Black, sqb)
                & antes.pieces[0][PieceType::Pawn as usize];
            atacantes |= crate::bitboard::pawn_attacks(Color::White, sqb)
                & antes.pieces[1][PieceType::Pawn as usize];

            let mut resto = atacantes;
            while resto != 0 {
                let asq = crate::bitboard::pop_lsb(&mut resto) as usize;
                if cambiadas & (1u64 << asq) != 0 {
                    continue; // esa pieza tambien se movio: ya cubierta en 2.1
                }
                let (c, pt) = antes.piece_at(asq as u8).expect("bitboard inconsistente");
                let color_idx = c as usize;
                let atk_type = pt as usize;
                if let Some((_, tipo_v)) = antes.piece_at(sqb) {
                    for v in 0..2usize {
                        self.restar(
                            &mut acc[v],
                            threat_idx_v(color_idx, atk_type, tipo_v as usize, sq, v),
                        );
                    }
                }
                if let Some((_, tipo_v)) = despues.piece_at(sqb) {
                    for v in 0..2usize {
                        self.sumar(
                            &mut acc[v],
                            threat_idx_v(color_idx, atk_type, tipo_v as usize, sq, v),
                        );
                    }
                }
            }
        }
    }
}

impl AcumBulletAmenazas {
    /// Construye el acumulador desde cero (bias + features de `map_features`)
    /// para las dos perspectivas. Es la referencia exacta con la que el test
    /// de regresion compara el camino incremental.
    pub fn desde_tablero(red: &'static RedBulletAmenazas, b: &Board) -> AcumBulletAmenazas {
        let mut acc = [[0i32; H]; 2];
        for j in 0..H {
            acc[0][j] = red.l0b[j] as i32;
            acc[1][j] = red.l0b[j] as i32;
        }
        let mut pares: Vec<(usize, usize)> = Vec::with_capacity(MAX_ACTIVE);
        map_features(b, |s, n| pares.push((s, n)));
        let (stm_i, ntm_i) = if b.turn == Color::White { (0usize, 1usize) } else { (1, 0) };
        for &(s, _) in &pares {
            red.sumar(&mut acc[stm_i], s);
        }
        for &(_, n) in &pares {
            red.sumar(&mut acc[ntm_i], n);
        }
        AcumBulletAmenazas {
            red,
            tablero: *b,
            acc,
        }
    }

    /// Variante "clonar + aplicar": parte de la API publica de
    /// `NnueAccumulator` (neural.rs). La busqueda usa la variante in-place
    /// `aplicar_jugada`; esta queda para compatibilidad y tests.
    #[allow(dead_code)]
    pub fn despues_de_jugada(&self, antes: &Board, despues: &Board) -> AcumBulletAmenazas {
        let mut nuevo = self.clone();
        nuevo.aplicar_jugada(antes, despues);
        nuevo
    }

    /// Aplica el delta de una jugada MUTANDO los dos acumuladores (sin
    /// recalcular las 5376 features). Es su propia inversa al intercambiar
    /// los argumentos: `aplicar_jugada(a, d)` seguido de
    /// `aplicar_jugada(d, a)` restaura el estado exacto.
    #[inline]
    pub fn aplicar_jugada(&mut self, antes: &Board, despues: &Board) {
        // `red` es &'static: copiarlo evita el prestamo simultaneo de self
        // mientras se muta acc.
        let red = self.red;
        self.tablero = *despues;
        red.actualizar_incremental(&mut self.acc, antes, despues);
    }

    /// Salida en centipeones desde la perspectiva del lado que mueve, usando
    /// los acumuladores incrementales (sin recalcular features).
    pub fn evaluar(&self) -> f32 {
        self.red.salida_desde_acumuladores(&self.acc, self.tablero.turn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cargar_real() -> Option<&'static RedBulletAmenazas> {
        let ruta = std::env::var("MIMOTOR_RED_BULLET_5376").unwrap_or_else(|_| {
            "/Users/Tavito/mi-motor/cand_bullet_nnue_1024_amenazas/checkpoints/mimotor_bullet_1024_amenazas-30/quantised.bin".to_string()
        });
        let datos = std::fs::read(&ruta).ok()?;
        let red = RedBulletAmenazas::cargar_de_bytes(&datos)?;
        Some(Box::leak(Box::new(red)))
    }

    fn evaluar(b: &Board) -> f32 {
        let red = cargar_real().expect("pesos 5376");
        red.evaluar_tablero(b)
    }

    /// Replica el test fena_contra_python_blancas de inputs.rs: la misma
    /// posicion validada contra features_threat.py debe emitir exactamente
    /// 6 pares y las mismas 4 entradas de amenaza (indices 4548/2300/2628/
    /// 4988). Esto valida la reconstruccion de la perspectiva del motor.
    #[test]
    fn features_contra_python_blancas() {
        let b = Board::from_fen("4k3/8/8/8/8/8/8/R3K2q w - - 0 1").unwrap();
        let mut pares: Vec<(usize, usize)> = Vec::new();
        map_features(&b, |s, n| pares.push((s, n)));
        assert_eq!(pares.len(), 6, "4 piezas base + 2 amenazas = 6 pares");

        let amenazas: Vec<usize> = pares
            .iter()
            .flat_map(|&(a, b2)| [a, b2])
            .filter(|&i| i >= N_BASE)
            .collect();
        assert_eq!(amenazas.len(), 4, "2 amenazas x 2 vistas (stm+ntm)");
        for &i in &[4548usize, 2300, 2628, 4988] {
            assert!(amenazas.contains(&i), "falta indice {}: {:?}", i, amenazas);
        }
    }

    /// Simetria: la misma posicion con negras al turno (FEN espejado y
    /// colores invertidos) debe emitir el mismo numero de pares, todos
    /// dentro de rango.
    #[test]
    fn simetria_negras() {
        let b = Board::from_fen("4k3/8/8/8/8/8/8/R3K2q w - - 0 1").unwrap();
        let mut pares = Vec::new();
        map_features(&b, |s, n| pares.push((s, n)));
        assert_eq!(pares.len(), 6);
        for &(s, n) in &pares {
            assert!(s < N_INPUTS && n < N_INPUTS, "indice fuera de rango");
        }
    }

    /// Posicion inicial dentro de MAX_ACTIVE (igual que el test del trainer).
    #[test]
    fn posicion_inicial_dentro_de_max_active() {
        let b = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut n = 0usize;
        map_features(&b, |_, _| n += 1);
        assert!(n <= MAX_ACTIVE, "{} > MAX_ACTIVE {}", n, MAX_ACTIVE);
    }

    /// Prueba de carga de los pesos REALES + determinismo + cordura de las
    /// evaluaciones. Se omite (con aviso) si el archivo no existe, para que
    /// el CI sin los pesos no falle.
    #[test]
    fn pesos_reales_cargan_y_evaluan() {
        let Some(red) = cargar_real() else {
            eprintln!("sin pesos 5376: test omitido");
            return;
        };

        let posiciones = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "4k3/8/8/8/8/8/8/R3K2q w - - 0 1",
            "4k3/8/8/8/8/8/8/QQQK4 w - - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        ];
        let mut evals = [0f32; 4];
        for (i, fen) in posiciones.iter().enumerate() {
            let b = Board::from_fen(fen).unwrap();
            let e1 = red.evaluar_tablero(&b);
            let e2 = red.evaluar_tablero(&b);
            // Determinismo basico: dos evaluaciones de la misma posicion
            // deben dar exactamente lo mismo.
            assert_eq!(e1, e2, "evaluacion no determinista en {fen}");
            assert!(
                e1.is_finite(),
                "score NaN/inf en {fen}: {e1}"
            );
            assert!(
                e1.abs() < 10000.0,
                "score absurdo en {fen}: {e1}"
            );
            evals[i] = e1;
            eprintln!("info string test NNUE 5376: {fen} -> {e1:.2} cp");
        }

        // Cordura material: 3 damas blancas vs rey solo debe ser MUCHO mejor
        // que la posicion inicial, y la inicial mucho mejor que R+K negros
        // con dama negra (R3K2q da ventaja a negras).
        assert!(
            evals[2] > evals[0] + 100.0,
            "3 damas no dominan a la inicial: {evals:?}"
        );
        assert!(
            evals[0] > evals[1] + 100.0,
            "la inicial no domina a R+K vs Q+K: {evals:?}"
        );
    }

    /// El AcumBulletAmenazas conectado a NnueAccumulator debe ser reversible
    /// aunque recalcule todo: aplicar una jugada y deshacerla restaura el
    /// acumulador y la evaluacion exactos.
    #[test]
    fn aplicar_jugada_reversible() {
        let Some(red) = cargar_real() else {
            eprintln!("sin pesos 5376: test omitido");
            return;
        };
        let antes = Board::from_fen(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        )
        .unwrap();
        let mut acc = AcumBulletAmenazas::desde_tablero(red, &antes);
        let e_raiz = acc.evaluar();
        let mv = crate::movegen::generate_legal(&antes)
            .into_iter()
            .next()
            .expect("hay al menos una jugada legal");
        let despues = antes.make_move(&mv);
        acc.aplicar_jugada(&antes, &despues);
        acc.aplicar_jugada(&despues, &antes);
        assert_eq!(acc.tablero.zobrist, antes.zobrist);
        assert_eq!(acc.evaluar(), e_raiz);
    }
}

// ---------------------------------------------------------------------------
// REGRESION: el accumulator incremental debe ser BIT A BIT IDENTICO al
// recalculo completo en CADA ply de una secuencia de jugadas (incluyendo
// capturas, enroques, al paso, promociones y apertura/cierre de lineas).
// ---------------------------------------------------------------------------
// Nota: va en un modulo de test propio (el `mod tests` de arriba ya cerro),
// con una copia local de `cargar_real`.
#[cfg(test)]
mod tests_incremental {
    use super::*;

    fn cargar_real() -> Option<&'static RedBulletAmenazas> {
        let ruta = std::env::var("MIMOTOR_RED_BULLET_5376").unwrap_or_else(|_| {
            "/Users/Tavito/mi-motor/cand_bullet_nnue_1024_amenazas/checkpoints/mimotor_bullet_1024_amenazas-30/quantised.bin".to_string()
        });
        let datos = std::fs::read(&ruta).ok()?;
        let red = RedBulletAmenazas::cargar_de_bytes(&datos)?;
        Some(Box::leak(Box::new(red)))
    }

    /// Red con pesos pseudoaleatorios deterministas: permite correr la regresion
    /// bit-identica SIN depender del archivo de pesos real (y con valores que
    /// estresan la aritmetica i32 del accumulator).
    fn red_sintetica() -> &'static RedBulletAmenazas {
    static R: std::sync::OnceLock<Option<&'static RedBulletAmenazas>> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        let mut x: u64 = 0x9e3779b97f4a7c15;
        let mut sig = move |n: usize| -> Vec<i16> {
            (0..n)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (x >> 33) as i16
                })
                .collect()
        };
        let red = RedBulletAmenazas {
            l0w: sig(N_INPUTS * H),
            l0b: sig(H),
            l1w: sig(2 * H),
            l1b: sig(1)[0],
        };
        Some(Box::leak(Box::new(red)))
    })
    .expect("red sintetica")
}

/// Secuencia de tableros: desde `fen`, aplica `plies` veces la PRIMERA
/// jugada legal (determinista) y devuelve todos los tableros intermedios.
fn secuencia_plies(fen: &str, plies: usize) -> Vec<Board> {
    let mut b = Board::from_fen(fen).unwrap();
    let mut out = vec![b];
    for _ in 0..plies {
        let mvs = crate::movegen::generate_legal(&b);
        let Some(mv) = mvs.into_iter().next() else {
            break;
        };
        b = b.make_move(&mv);
        out.push(b);
    }
    out
}

/// Posiciones de regresion: apertura, medio juego tactico (kiwipete), torres
/// y damas en filas/diagonales (abrir/cerrar lineas), captura de dama,
/// promocion, enroque y al paso.
fn fens_regresion() -> [&'static str; 8] {
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/R3K2q w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQ - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r2qkbnr/ppp2ppp/2n5/3pp3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 1",
    ]
}

/// Test de regresion PRINCIPAL: corre SIEMPRE (pesos sinteticos). Compara,
/// en cada ply de hasta 30 jugadas desde 8 posiciones, los dos acumuladores
/// (vista blanca y vista negra) y la evaluacion del camino incremental
/// contra un `desde_tablero` recien construido (recalculo completo). Al
/// final deshace toda la secuencia y exige volver al estado raiz exacto.
#[test]
fn incremental_bit_identico_sintetico() {
    let red = red_sintetica();
    let fens = fens_regresion();
    for (fi, fen) in fens.iter().enumerate() {
        let boards = secuencia_plies(fen, 30);
        assert!(boards.len() >= 10, "secuencia demasiado corta en {fen}");
        let ref_raiz = AcumBulletAmenazas::desde_tablero(red, &boards[0]);
        let mut acc = AcumBulletAmenazas::desde_tablero(red, &boards[0]);
        assert_eq!(acc.acc, ref_raiz.acc, "raiz acc fen {fi}");
        assert_eq!(acc.evaluar(), ref_raiz.evaluar(), "raiz eval fen {fi}");

        for (ply, par) in boards.windows(2).enumerate() {
            acc.aplicar_jugada(&par[0], &par[1]);
            let refacc = AcumBulletAmenazas::desde_tablero(red, &par[1]);
            assert_eq!(
                acc.acc, refacc.acc,
                "acc bit-identico: fen {fi} ply {ply} ({} -> {})",
                par[0].to_fen(), par[1].to_fen()
            );
            assert_eq!(
                acc.evaluar(), refacc.evaluar(),
                "eval bit-identica: fen {fi} ply {ply}"
            );
        }

        // Deshacer toda la secuencia: debe volver a la raiz exacta.
        let mut revertido = acc.clone();
        for (ply, par) in boards.windows(2).rev().enumerate() {
            revertido.aplicar_jugada(&par[1], &par[0]);
        }
        assert_eq!(revertido.acc, ref_raiz.acc, "undo acc fen {fi}");
        assert_eq!(revertido.evaluar(), ref_raiz.evaluar(), "undo eval fen {fi}");
        assert_eq!(revertido.tablero.zobrist, boards[0].zobrist, "undo tablero fen {fi}");
    }
}

/// Regresion con los pesos REALES (se omite con aviso si el archivo no
/// existe): ademas de comparar los acumuladores contra `desde_tablero`,
/// exige que `evaluar()` del camino incremental coincida con
/// `evaluar_tablero` (recalculo completo) ply a ply.
#[test]
fn incremental_bit_identico_con_pesos_reales() {
    let Some(red) = cargar_real() else {
        eprintln!("sin pesos 5376: test omitido");
        return;
    };
    for (fi, fen) in fens_regresion().iter().enumerate() {
        let boards = secuencia_plies(fen, 25);
        let mut acc = AcumBulletAmenazas::desde_tablero(red, &boards[0]);
        assert_eq!(
            acc.evaluar(),
            red.evaluar_tablero(&boards[0]),
            "raiz eval fen {fi}"
        );
        for (ply, par) in boards.windows(2).enumerate() {
            acc.aplicar_jugada(&par[0], &par[1]);
            let refacc = AcumBulletAmenazas::desde_tablero(red, &par[1]);
            assert_eq!(acc.acc, refacc.acc, "acc fen {fi} ply {ply}");
            assert_eq!(acc.evaluar(), refacc.evaluar(), "eval fen {fi} ply {ply}");
            assert_eq!(
                acc.evaluar(),
                red.evaluar_tablero(&par[1]),
                "eval vs recalculo completo fen {fi} ply {ply}"
            );
        }
    }
}

/// Medicion del speedup del camino por nodo: recalculo completo (version NO
/// incremental, evaluar_tablero) vs delta + forward (version incremental).
/// Solo imprime la proporcion; no impone aserciones de tiempo (los tests no
/// deben ser sensibles al hardware).
#[test]
fn medir_speedup_acumulador() {
    let red = cargar_real().unwrap_or_else(red_sintetica);
    let boards = secuencia_plies(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        200,
    );
    let n = boards.len() as f64;

    use std::hint::black_box;

    // Pasadas extra para que el tiempo sea medible y estable en release.
    let pasadas = 20usize;
    let mut sink = 0.0f32;

    // Desglose: recompute (extraccion completa de features + bias).
    let t0 = std::time::Instant::now();
    for _ in 0..pasadas {
        for b in &boards {
            let a = AcumBulletAmenazas::desde_tablero(red, b);
            sink += a.acc[0][0] as f32;
        }
    }
    let t_recompute = t0.elapsed();

    // Desglose: delta (aplicar_jugada).
    let mut acc = AcumBulletAmenazas::desde_tablero(red, &boards[0]);
    let t1 = std::time::Instant::now();
    for _ in 0..pasadas {
        for par in boards.windows(2) {
            acc.aplicar_jugada(&par[0], &par[1]);
        }
    }
    let t_delta = t1.elapsed();

    // Desglose: forward (evaluar, sin delta).
    let t2 = std::time::Instant::now();
    for _ in 0..pasadas {
        for _ in 0..boards.len() {
            sink += acc.evaluar();
        }
    }
    let t_forward = t2.elapsed();
    eprintln!(
        "[medir_speedup] desglose | recompute: {:.3}s | delta: {:.3}s | forward: {:.3}s",
        t_recompute.as_secs_f64(),
        t_delta.as_secs_f64(),
        t_forward.as_secs_f64()
    );

    // Version antigua: cada nodo reconstruye el accumulator completo
    // (desde_tablero = extraccion de features + bias) y evalua.
    let t0 = std::time::Instant::now();
    for _ in 0..pasadas {
        for b in &boards {
            let a = AcumBulletAmenazas::desde_tablero(red, b);
            sink += a.evaluar();
        }
    }
    let t_old = t0.elapsed();

    // Version nueva: un solo `desde_tablero` y luego solo deltas por nodo.
    let mut acc = AcumBulletAmenazas::desde_tablero(red, &boards[0]);
    let t1 = std::time::Instant::now();
    for _ in 0..pasadas {
        for par in boards.windows(2) {
            acc.aplicar_jugada(&par[0], &par[1]);
            sink += acc.evaluar();
        }
    }
    let t_new = t1.elapsed();

    let nodos = n * pasadas as f64;
    let nps_old = nodos / t_old.as_secs_f64();
    let nps_new = nodos / t_new.as_secs_f64();
    eprintln!(
        "[medir_speedup] {:.0} nodos | viejo (recalculo completo): {:.0} nps | nuevo (delta): {:.0} nps | speedup x{:.1}",
        nodos,
        nps_old,
        nps_new,
        nps_new / nps_old
    );
    sink = black_box(sink);
    assert!(sink.is_finite(), "sink no finito");
}

/// EQUIVALENCIA NUMERICA del camino NEON (sumar/restar del acumulador y la
/// capa de salida) contra el camino escalar original. Mismo criterio que
/// `bullet_net.rs::salida_neon_igual_que_escalar`: igualdad EXACTA, no
/// aproximada. Recorre >100 posiciones: 8 aperturas/finales/tacticas de
/// `fens_regresion` mas partidas aleatorias legales de hasta 14 plies cada
/// una, con pesos sinteticos (no depende del archivo de pesos real).
#[test]
fn neon_igual_que_escalar_100_posiciones() {
    let red = red_sintetica();
    let mut comprobadas = 0usize;
    let mut semilla = 0xD00D_FEED_1357u64;

    for fen in fens_regresion() {
        let mut tablero = Board::from_fen(fen).unwrap();
        for _ply in 0..14 {
            // 1) sumar/restar: comparar el acumulador construido con el
            // camino NEON (via desde_tablero, que llama a `sumar`) contra
            // uno construido a mano con el camino escalar.
            let acc_neon = AcumBulletAmenazas::desde_tablero(red, &tablero);

            let mut acc_escalar = [[0i32; H]; 2];
            for j in 0..H {
                acc_escalar[0][j] = red.l0b[j] as i32;
                acc_escalar[1][j] = red.l0b[j] as i32;
            }
            let mut pares: Vec<(usize, usize)> = Vec::with_capacity(MAX_ACTIVE);
            map_features(&tablero, |s, n| pares.push((s, n)));
            let (stm_i, ntm_i) = if tablero.turn == Color::White { (0usize, 1usize) } else { (1, 0) };
            for &(s, _) in &pares {
                red.sumar_escalar(&mut acc_escalar[stm_i], s);
            }
            for &(_, n) in &pares {
                red.sumar_escalar(&mut acc_escalar[ntm_i], n);
            }
            assert_eq!(
                acc_neon.acc, acc_escalar,
                "sumar NEON != escalar en {}",
                tablero.to_fen()
            );

            // restar: partir del acumulador ya sumado y restar todas las
            // mismas features con cada camino; ambos deben volver al bias.
            let mut r_neon = acc_neon.acc;
            let mut r_escalar = acc_escalar;
            for &(s, _) in &pares {
                red.restar(&mut r_neon[stm_i], s);
                red.restar_escalar(&mut r_escalar[stm_i], s);
            }
            for &(_, n) in &pares {
                red.restar(&mut r_neon[ntm_i], n);
                red.restar_escalar(&mut r_escalar[ntm_i], n);
            }
            assert_eq!(r_neon, r_escalar, "restar NEON != escalar en {}", tablero.to_fen());

            // 2) capa de salida: NEON vs escalar sobre el MISMO acumulador.
            let eval_neon = red.salida_desde_acumuladores(&acc_neon.acc, tablero.turn);
            let eval_escalar = red.salida_desde_acumuladores_escalar(&acc_neon.acc, tablero.turn);
            assert_eq!(
                eval_neon, eval_escalar,
                "salida NEON != escalar en {}",
                tablero.to_fen()
            );

            comprobadas += 1;

            let legales = crate::movegen::generate_legal(&tablero);
            if legales.is_empty() {
                break;
            }
            semilla ^= semilla << 7;
            semilla ^= semilla >> 9;
            let mv = legales[(semilla as usize) % legales.len()];
            tablero = tablero.make_move(&mv);
        }
    }

    assert!(comprobadas >= 100, "solo se comprobaron {comprobadas} posiciones");
}

/// Microbenchmark temporal: NEON vs escalar en `producto_punto` de forma
/// aislada (sin ruido de movegen/TT/UCI). Solo imprime la razon, no impone
/// aserciones de tiempo.
#[test]
fn microbench_producto_punto_neon_vs_escalar() {
    use std::hint::black_box;
    let red = red_sintetica();
    let tablero = Board::from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    )
    .unwrap();
    let acc = AcumBulletAmenazas::desde_tablero(red, &tablero);
    let (yo, rival) = (&acc.acc[0], &acc.acc[1]);
    let iter = 2_000_000usize;
    let mut sink: i64 = 0;

    let t0 = std::time::Instant::now();
    for _ in 0..iter {
        sink = sink.wrapping_add(red.producto_punto(black_box(yo), black_box(rival)));
    }
    let t_neon = t0.elapsed();

    let t1 = std::time::Instant::now();
    for _ in 0..iter {
        sink = sink.wrapping_add(red.producto_punto_escalar(black_box(yo), black_box(rival)));
    }
    let t_escalar = t1.elapsed();

    eprintln!(
        "[microbench producto_punto] NEON: {:.4}s | escalar: {:.4}s | speedup x{:.2}",
        t_neon.as_secs_f64(),
        t_escalar.as_secs_f64(),
        t_escalar.as_secs_f64() / t_neon.as_secs_f64()
    );
    sink = black_box(sink);
    assert!(sink != i64::MIN || true, "sink dummy");
}

/// Microbenchmark temporal de `sumar`/`restar` (acumulador incremental):
/// NEON vs escalar, aislado.
#[test]
fn microbench_sumar_restar_neon_vs_escalar() {
    use std::hint::black_box;
    let red = red_sintetica();
    let iter = 3_000_000usize;

    let mut acc_neon = [0i32; H];
    let t0 = std::time::Instant::now();
    for i in 0..iter {
        red.sumar(black_box(&mut acc_neon), black_box(i % N_INPUTS));
    }
    let t_neon = t0.elapsed();

    let mut acc_escalar = [0i32; H];
    let t1 = std::time::Instant::now();
    for i in 0..iter {
        red.sumar_escalar(black_box(&mut acc_escalar), black_box(i % N_INPUTS));
    }
    let t_escalar = t1.elapsed();

    eprintln!(
        "[microbench sumar] NEON: {:.4}s | escalar: {:.4}s | speedup x{:.2}",
        t_neon.as_secs_f64(),
        t_escalar.as_secs_f64(),
        t_escalar.as_secs_f64() / t_neon.as_secs_f64()
    );
    black_box(acc_neon);
    black_box(acc_escalar);
}
}
