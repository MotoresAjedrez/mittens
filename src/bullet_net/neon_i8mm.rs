//! Salida SCReLU exacta con NEON/I8MM para ARM que habilita esa extension.
//! Se separa cada cuadrado u16 en dos bytes: v² = low + 256*high.
//! USDOT permite multiplicar esos bytes sin signo por los pesos i8 con signo.
//! La red y su escala no cambian; los pesos que no caben en i8 usan el fallback.
use super::*;
use std::arch::aarch64::*;

/// Determina si debe usarse el kernel USDOT (`FEAT_I8MM`).
/// Por defecto se activa cuando el binario se compila con `target_feature = "i8mm"`
/// (ej. `apple-m2`..`apple-m5` o Android con `+i8mm`), donde `actualizar_y_salida_i8mm`
/// se integra `#[inline(always)]` sin sobrecosto de llamada ABI. En compilaciones
/// sin `+i8mm` global (ej. `apple-m1` portable) puede forzarse con `MITTENS_I8MM=1`
/// si el procesador soporta `i8mm`.
#[inline(always)]
pub(super) fn tiene_i8mm() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let v = CACHE.load(Ordering::Relaxed);
    if v != 0 {
        return v == 2;
    }
    #[cold]
    fn inicializar() -> bool {
        let usar = match std::env::var("MITTENS_I8MM").ok().as_deref().map(str::trim) {
            Some("0" | "false" | "no" | "off") => false,
            Some("1" | "true" | "yes" | "on") => {
                #[cfg(target_feature = "i8mm")]
                {
                    true
                }
                #[cfg(not(target_feature = "i8mm"))]
                {
                    std::arch::is_aarch64_feature_detected!("i8mm")
                }
            }
            _ => cfg!(target_feature = "i8mm"),
        };
        CACHE.store(if usar { 2 } else { 1 }, Ordering::Relaxed);
        usar
    }
    inicializar()
}

// El intrinseco Rust vusdotq_s32 sigue siendo experimental. Esta instruccion
// se emite en Rust estable dentro de funciones con #[target_feature(enable = "i8mm")].
// Definido como macro para garantizar inlining completo incluso cuando el binario
// base se compila sin +i8mm global (ej. target-cpu=apple-m1 o Android +dotprod).
macro_rules! usdot {
    ($a:expr, $b:expr, $c:expr) => {{
        let mut a_reg: int32x4_t = $a;
        let b_reg: uint8x16_t = $b;
        let c_reg: int8x16_t = $c;
        core::arch::asm!(
            "usdot {a:v}.4s, {b:v}.16b, {c:v}.16b",
            a = inout(vreg) a_reg,
            b = in(vreg) b_reg,
            c = in(vreg) c_reg,
            options(pure, nomem, nostack, preserves_flags)
        );
        a_reg
    }};
}

macro_rules! acumular {
    ($a:expr, $b:expr, $w:expr, $lo:ident, $hi:ident) => {{
        // SQXTUN hace exactamente clamp(i16,0,255) al estrechar a u8.
        let v = vqmovun_high_s16(vqmovun_s16($a), $b);
        let s0 = vreinterpretq_u8_u16(vmull_u8(vget_low_u8(v), vget_low_u8(v)));
        let s1 = vreinterpretq_u8_u16(vmull_high_u8(v, v));
        // En little-endian, los bytes pares de cada u16 son v² mod 256 (low) y
        // los impares son v² >> 8 (high). UZP1/UZP2 los separan en 2 instrucciones
        // independientes en vez de 4 (XTN + XTN2 + SHRN + SHRN2).
        let low = vuzp1q_u8(s0, s1);
        let high = vuzp2q_u8(s0, s1);
        let w_vec = vld1q_s8($w);
        $lo = usdot!($lo, low, w_vec);
        $hi = usdot!($hi, high, w_vec);
    }};
}

impl RedBullet {
    #[cfg_attr(not(target_feature = "i8mm"), target_feature(enable = "i8mm"))]
    #[cfg_attr(target_feature = "i8mm", inline(always))]
    pub(super) unsafe fn actualizar_y_salida_i8mm(
        &self,
        src: &[[i16; H_MAX]; 2],
        dst: &mut [[i16; H_MAX]; 2],
        cambios: &[Cambios; 2],
        negras_mueven: bool,
        bucket: usize,
    ) -> Option<i64> {
        unsafe {
            use std::arch::aarch64::*;
            let forma = (
                cambios[0].add.len(),
                cambios[0].sub.len(),
                cambios[1].add.len(),
                cambios[1].sub.len(),
            );
            if !matches!(forma, (1, 1, 1, 1) | (1, 2, 1, 2) | (2, 2, 2, 2)) {
                return None;
            }
            let h = self.h;
            let col = |f: u16| self.l0w.as_ptr().add(f as usize * h);
            let w = self.l1w_i8.as_ref()?.as_ptr().add(bucket * 2 * H_MAX);
            let (ws0, ws1) = if negras_mueven {
                (w.add(H_MAX), w)
            } else {
                (w, w.add(H_MAX))
            };
            let (mut lo0, mut hi0, mut lo1, mut hi1) = (
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
            );
            macro_rules! combinar {
                (+ $a:expr,$b:expr) => {
                    vaddq_s16($a, $b)
                };
                (- $a:expr,$b:expr) => {
                    vsubq_s16($a, $b)
                };
            }
            macro_rules! bloque {
                ($t:expr,$p:expr,$ws:expr,[$($signo:tt $col:expr),+])=>{{
                    let s=src[$p].as_ptr().add($t);
                    let mut r0=vld1q_s16(s);
                    let mut r1=vld1q_s16(s.add(8));
                    let mut r2=vld1q_s16(s.add(16));
                    let mut r3=vld1q_s16(s.add(24));
                    $(
                        let w=$col.add($t);
                        r0=combinar!($signo r0,vld1q_s16(w));
                        r1=combinar!($signo r1,vld1q_s16(w.add(8)));
                        r2=combinar!($signo r2,vld1q_s16(w.add(16)));
                        r3=combinar!($signo r3,vld1q_s16(w.add(24)));
                    )+
                    let d=dst[$p].as_mut_ptr().add($t);
                    vst1q_s16(d,r0);
                    vst1q_s16(d.add(8),r1);
                    vst1q_s16(d.add(16),r2);
                    vst1q_s16(d.add(24),r3);
                    acumular!(r0,r1,$ws.add($t),lo0,hi0);
                    acumular!(r2,r3,$ws.add($t+16),lo1,hi1);
                }};
            }
            macro_rules! pasada {
                ([$($s0:tt $c0:expr),+],[$($s1:tt $c1:expr),+])=>{{
                    for t in (0..h).step_by(32) {
                        bloque!(t,0,ws0,[$($s0 $c0),+]);
                        bloque!(t,1,ws1,[$($s1 $c1),+]);
                    }
                }};
            }
            let (c0, c1) = (&cambios[0], &cambios[1]);
            match forma {
                (1, 1, 1, 1) => {
                    pasada!([+ col(c0.add[0]),- col(c0.sub[0])],[+ col(c1.add[0]),- col(c1.sub[0])])
                }
                (1, 2, 1, 2) => {
                    pasada!([+ col(c0.add[0]),- col(c0.sub[0]),- col(c0.sub[1])],[+ col(c1.add[0]),- col(c1.sub[0]),- col(c1.sub[1])])
                }
                (2, 2, 2, 2) => {
                    pasada!([+ col(c0.add[0]),+ col(c0.add[1]),- col(c0.sub[0]),- col(c0.sub[1])],[+ col(c1.add[0]),+ col(c1.add[1]),- col(c1.sub[0]),- col(c1.sub[1])])
                }
                _ => unreachable!(),
            }
            Some(vaddlvq_s32(vaddq_s32(lo0, lo1)) + (vaddlvq_s32(vaddq_s32(hi0, hi1)) << 8))
        }
    }

    #[cfg_attr(not(target_feature = "i8mm"), target_feature(enable = "i8mm"))]
    #[cfg_attr(target_feature = "i8mm", inline(always))]
    pub(super) unsafe fn producto_punto_i8mm(
        &self,
        yo: &[i16; H_MAX],
        rival: &[i16; H_MAX],
        bucket: usize,
    ) -> Option<i64> {
        unsafe {
            let w = self.l1w_i8.as_ref()?.as_ptr().add(bucket * 2 * H_MAX);
            let (mut lo0, mut hi0, mut lo1, mut hi1) = (
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
            );
            for t in (0..self.h).step_by(16) {
                acumular!(
                    vld1q_s16(yo.as_ptr().add(t)),
                    vld1q_s16(yo.as_ptr().add(t + 8)),
                    w.add(t),
                    lo0,
                    hi0
                );
                acumular!(
                    vld1q_s16(rival.as_ptr().add(t)),
                    vld1q_s16(rival.as_ptr().add(t + 8)),
                    w.add(H_MAX + t),
                    lo1,
                    hi1
                );
            }
            // Cada suma de bytes recibe 2*h/4 <= 512 terminos por carril,
            // cada uno de magnitud <=255*128: 16_711_680 < i32::MAX.
            // Ensanchar a i64 ANTES de escalar los bytes altos por 256.
            Some(vaddlvq_s32(vaddq_s32(lo0, lo1)) + (vaddlvq_s32(vaddq_s32(hi0, hi1)) << 8))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i8mm_salida_y_fusion_exactas_formatos_buckets_y_extremos() {
        for (h, buckets) in ARQUITECTURAS_SOPORTADAS {
            for patron in 0..3 {
                let mut bytes = vec![0u8; bytes_utiles(h, buckets)];
                let inicio = (N_ENTRADA * h + h) * 2;
                for (i, par) in bytes[inicio..inicio + buckets * 2 * h * 2]
                    .chunks_exact_mut(2)
                    .enumerate()
                {
                    let w: i16 = match patron {
                        0 => 127,
                        1 => -128,
                        _ => ((i * 73) % 256) as i16 - 128,
                    };
                    par.copy_from_slice(&w.to_le_bytes());
                }
                let mut red = RedBullet::cargar_de_bytes(&bytes).unwrap();
                assert!(red.l1w_i8.is_some());
                if patron == 2 {
                    for (i, w) in red.l0w.iter_mut().enumerate() {
                        *w = (i.wrapping_mul(7919) as i16).wrapping_sub(16000);
                    }
                }
                for valor in [255i16, i16::MIN, i16::MAX, -1, 0] {
                    let src = [[valor; H_MAX]; 2];
                    for (adds, subs) in [(1, 1), (1, 2), (2, 2)] {
                        let cambios = std::array::from_fn(|p| {
                            let mut c = Cambios::default();
                            c.add.extend((0..adds).map(|i| (3 + p * 7 + i) as u16));
                            c.sub.extend((0..subs).map(|i| (20 + p * 7 + i) as u16));
                            c
                        });
                        let mut esperado = [[12345i16; H_MAX]; 2];
                        for p in 0..2 {
                            for j in 0..h {
                                let mut v = src[p][j];
                                for &f in &cambios[p].add {
                                    v = v.wrapping_add(red.l0w[f as usize * h + j]);
                                }
                                for &f in &cambios[p].sub {
                                    v = v.wrapping_sub(red.l0w[f as usize * h + j]);
                                }
                                esperado[p][j] = v;
                            }
                        }
                        for negras in [false, true] {
                            for bucket in 0..buckets {
                                let mut dst = [[12345i16; H_MAX]; 2];
                                let suma = unsafe {
                                    red.actualizar_y_salida_i8mm(
                                        &src, &mut dst, &cambios, negras, bucket,
                                    )
                                }
                                .unwrap();
                                assert_eq!(dst, esperado);
                                let mut referencia = 0i64;
                                for p in 0..2 {
                                    let mitad = if (p == 1) == negras { 0 } else { H_MAX };
                                    for j in 0..h {
                                        let v = esperado[p][j].clamp(0, QA as i16) as i64;
                                        referencia +=
                                            v * v * red.l1w[bucket * 2 * H_MAX + mitad + j] as i64;
                                    }
                                }
                                assert_eq!(
                                    suma, referencia,
                                    "h={h} bk={bucket} patron={patron} valor={valor} negras={negras}"
                                );
                                let (yo, rival) = if negras {
                                    (&esperado[1], &esperado[0])
                                } else {
                                    (&esperado[0], &esperado[1])
                                };
                                assert_eq!(
                                    unsafe { red.producto_punto_i8mm(yo, rival, bucket) },
                                    Some(referencia)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn i8mm_rechaza_pesos_no_representables_y_formas_raras_sin_escribir() {
        for peso in [-32768i16, -129, 128, 129, 32767] {
            let mut bytes = vec![0u8; bytes_utiles(1024, 8)];
            let offset = (N_ENTRADA * 1024 + 1024) * 2;
            bytes[offset..offset + 2].copy_from_slice(&peso.to_le_bytes());
            let red = RedBullet::cargar_de_bytes(&bytes).unwrap();
            assert!(red.l1w_i8.is_none(), "peso={peso}");
        }
        let red = RedBullet::cargar_de_bytes(&vec![0; bytes_utiles(1024, 8)]).unwrap();
        let src = [[0; H_MAX]; 2];
        let mut dst = [[23456; H_MAX]; 2];
        let cambios = [Cambios::default(), Cambios::default()];
        assert!(
            unsafe { red.actualizar_y_salida_i8mm(&src, &mut dst, &cambios, false, 0) }.is_none()
        );
        assert_eq!(dst, [[23456; H_MAX]; 2]);
    }

    #[test]
    fn i8mm_bit_exacta_contra_neon_y_escalar_en_red_real() {
        let datos = include_bytes!("../../pesos_bullet_1024_buckets8.bin");
        let red_i8mm = Box::leak(Box::new(RedBullet::cargar_de_bytes(datos).unwrap()));
        assert!(
            red_i8mm.l1w_i8.is_some(),
            "la red real embebida debe tener todos sus pesos l1w en i8 (-128..=127)"
        );
        let mut red_neon_clon = RedBullet::cargar_de_bytes(datos).unwrap();
        red_neon_clon.l1w_i8 = None; // Fuerza el camino NEON estandar SMLAL
        let red_neon = Box::leak(Box::new(red_neon_clon));

        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
        ];
        let mut semilla = 0xCAFE_BABE_1357_2468u64;
        let mut verificadas = 0usize;
        for fen in fens {
            let raiz = crate::board::Board::from_fen(fen).unwrap();
            let mut ac_i8mm = AcumBullet::desde_tablero(red_i8mm, &raiz);
            let mut ac_neon = AcumBullet::desde_tablero(red_neon, &raiz);
            let mut actual = raiz;
            for _ply in 0..30 {
                assert_eq!(ac_i8mm.evaluar().to_bits(), ac_neon.evaluar().to_bits());
                assert_eq!(ac_i8mm.evaluar().to_bits(), ac_i8mm.evaluar_escalar().to_bits());
                assert_eq!(ac_i8mm.persp(), ac_neon.persp());
                verificadas += 1;

                let legales = crate::movegen::generate_legal(&actual);
                if legales.is_empty() {
                    break;
                }
                semilla ^= semilla << 13;
                semilla ^= semilla >> 7;
                semilla ^= semilla << 17;
                let mv = legales[(semilla as usize) % legales.len()];
                let hijo = actual.make_move(&mv);
                ac_i8mm.entrar(&actual, &hijo);
                ac_neon.entrar(&actual, &hijo);
                actual = hijo;
            }
        }
        assert!(verificadas >= 120);
    }
}
