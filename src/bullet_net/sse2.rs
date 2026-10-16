//! Kernels enteros de 128 bits para las compilaciones portables SSE2 de la NNUE Bullet.
//! Las cargas son unaligned: Vec<i16> no garantiza alineacion de 32 bytes.

use super::*;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn suma_i64(v: __m128i) -> i64 {
    unsafe {
        let mut carriles = [0i32; 4];
        _mm_storeu_si128(carriles.as_mut_ptr().cast(), v);
        carriles.into_iter().map(i64::from).sum()
    }
}

/// clamp(v)*w cabe en i16 para |w|<=128; PMADDWD calcula (v*w)*v.
#[inline(always)]
unsafe fn screlu(v: __m128i, w: *const i16) -> __m128i {
    unsafe {
        let c = _mm_min_epi16(
            _mm_max_epi16(v, _mm_setzero_si128()),
            _mm_set1_epi16(QA as i16),
        );
        let cw = _mm_mullo_epi16(c, _mm_loadu_si128(w.cast()));
        _mm_madd_epi16(cw, c)
    }
}

impl RedBullet {
    /// SSE2 disponible en x86-64: h<=1024, h%32==0,
    /// pesos de salida en -128..=128, features validas y bucket existente.
    #[inline(always)]
    pub(super) unsafe fn actualizar_y_salida(
        &self,
        src: &[[i16; H_MAX]; 2],
        dst: &mut [[i16; H_MAX]; 2],
        cambios: &[Cambios; 2],
        negras_mueven: bool,
        bucket: usize,
    ) -> Option<i64> {
        unsafe {
            // Rechazar antes de escribir: las formas raras usan el fallback.
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
            let ws = self.l1w.as_ptr().add(bucket * 2 * H_MAX);
            let (ws0, ws1) = if negras_mueven {
                (ws.add(H_MAX), ws)
            } else {
                (ws, ws.add(H_MAX))
            };
            let (s0, s1) = (src[0].as_ptr(), src[1].as_ptr());
            let (d0, d1) = (dst[0].as_mut_ptr(), dst[1].as_mut_ptr());
            let (mut a0, mut a1) = (_mm_setzero_si128(), _mm_setzero_si128());
            macro_rules! combinar {
                (+ $a:expr, $b:expr) => {
                    _mm_add_epi16($a, $b)
                };
                (- $a:expr, $b:expr) => {
                    _mm_sub_epi16($a, $b)
                };
            }
            macro_rules! bloque {
                ($t:expr, $s:expr, $d:expr, $w:expr, $a:ident, [$($signo:tt $col:expr),+]) => {{
                    let mut x = _mm_loadu_si128($s.add($t).cast());
                    let mut y = _mm_loadu_si128($s.add($t + 8).cast());
                    $(
                        let w = $col.add($t);
                        x = combinar!($signo x, _mm_loadu_si128(w.cast()));
                        y = combinar!($signo y, _mm_loadu_si128(w.add(8).cast()));
                    )+
                    _mm_storeu_si128($d.add($t).cast(), x);
                    _mm_storeu_si128($d.add($t + 8).cast(), y);
                    $a = _mm_add_epi32($a, screlu(x, $w.add($t)));
                    $a = _mm_add_epi32($a, screlu(y, $w.add($t + 8)));
                }};
            }
            macro_rules! pasada {
                ([$($s0:tt $c0:expr),+], [$($s1:tt $c1:expr),+]) => {{
                    for t in (0..h).step_by(16) {
                        bloque!(t, s0, d0, ws0, a0, [$($s0 $c0),+]);
                        bloque!(t, s1, d1, ws1, a1, [$($s1 $c1),+]);
                    }
                }};
            }
            let (c0, c1) = (&cambios[0], &cambios[1]);
            match forma {
                (1, 1, 1, 1) => pasada!([+ col(c0.add[0]), - col(c0.sub[0])],
                                       [+ col(c1.add[0]), - col(c1.sub[0])]),
                (1, 2, 1, 2) => pasada!([+ col(c0.add[0]), - col(c0.sub[0]), - col(c0.sub[1])],
                                       [+ col(c1.add[0]), - col(c1.sub[0]), - col(c1.sub[1])]),
                (2, 2, 2, 2) => {
                    pasada!([+ col(c0.add[0]), + col(c0.add[1]), - col(c0.sub[0]), - col(c0.sub[1])],
                                       [+ col(c1.add[0]), + col(c1.add[1]), - col(c1.sub[0]), - col(c1.sub[1])])
                }
                _ => unreachable!(),
            }
            // Por perspectiva/carril: 1024/4 terminos de como mucho
            // 65025*128 -> 2,130,739,200 < i32::MAX. Reunir en i64.
            Some(suma_i64(a0) + suma_i64(a1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_sse2_extremos_formatos_y_buckets() {
        for (h, buckets) in ARQUITECTURAS_SOPORTADAS {
            let mut red = RedBullet::cargar_de_bytes(&vec![0; bytes_utiles(h, buckets)]).unwrap();
            for patron in 0..3 {
                for (i, w) in red.l0w.iter_mut().enumerate() {
                    *w = if patron == 0 {
                        0
                    } else {
                        (i.wrapping_mul(7919) as i16).wrapping_sub(16000)
                    };
                }
                for (i, w) in red.l1w.iter_mut().enumerate() {
                    *w = match patron {
                        0 => 128,
                        1 => -128,
                        _ => ((i * 73) % 257) as i16 - 128,
                    };
                }
                for valor in [255i16, i16::MIN, i16::MAX] {
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
                                    red.actualizar_y_salida(
                                        &src, &mut dst, &cambios, negras, bucket,
                                    )
                                }
                                .expect("forma comun debe usar el kernel SSE2");
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
                                    "h={h} buckets={buckets} patron={patron} valor={valor} negras={negras}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fusion_sse2_forma_rara_no_escribe_destino() {
        let red = RedBullet::cargar_de_bytes(&vec![0; bytes_utiles(1024, 8)]).unwrap();
        let src = [[0; H_MAX]; 2];
        let mut dst = [[23456; H_MAX]; 2];
        let cambios = [Cambios::default(), Cambios::default()];
        assert!(unsafe { red.actualizar_y_salida(&src, &mut dst, &cambios, false, 0) }.is_none());
        assert_eq!(dst, [[23456; H_MAX]; 2]);
    }
}
