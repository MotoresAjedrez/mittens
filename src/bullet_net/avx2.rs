//! Kernels enteros de 256 bits para las compilaciones AVX2 de la NNUE Bullet.
//! Las cargas son unaligned: Vec<i16> no garantiza alineacion de 32 bytes.

use super::*;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn suma_i64(v: __m256i) -> i64 {
    unsafe {
        let mut carriles = [0i32; 8];
        _mm256_storeu_si256(carriles.as_mut_ptr().cast(), v);
        carriles.into_iter().map(i64::from).sum()
    }
}

/// clamp(v)*w cabe en i16 para |w|<=128; PMADDWD calcula (v*w)*v.
#[inline(always)]
unsafe fn screlu(v: __m256i, w: *const i16) -> __m256i {
    unsafe {
        let c = _mm256_min_epi16(
            _mm256_max_epi16(v, _mm256_setzero_si256()),
            _mm256_set1_epi16(QA as i16),
        );
        let cw = _mm256_mullo_epi16(c, _mm256_loadu_si256(w.cast()));
        _mm256_madd_epi16(cw, c)
    }
}

impl RedBullet {
    /// src y dst pueden coincidir. Se lee cada bloque antes de escribirlo.
    #[inline(always)]
    unsafe fn actualizar_avx2(&self, src: *const i16, dst: *mut i16, add: &[u16], sub: &[u16]) {
        unsafe {
            let h = self.h;
            let col = |f: u16| self.l0w.as_ptr().add(f as usize * h);
            macro_rules! combinar {
                (+ $a:expr, $b:expr) => {
                    _mm256_add_epi16($a, $b)
                };
                (- $a:expr, $b:expr) => {
                    _mm256_sub_epi16($a, $b)
                };
            }
            macro_rules! pasada {
                ($($signo:tt $col:expr),+ $(,)?) => {{
                    for t in (0..h).step_by(32) {
                        let mut a = _mm256_loadu_si256(src.add(t).cast());
                        let mut b = _mm256_loadu_si256(src.add(t + 16).cast());
                        $(
                            let w = $col.add(t);
                            a = combinar!($signo a, _mm256_loadu_si256(w.cast()));
                            b = combinar!($signo b, _mm256_loadu_si256(w.add(16).cast()));
                        )+
                        _mm256_storeu_si256(dst.add(t).cast(), a);
                        _mm256_storeu_si256(dst.add(t + 16).cast(), b);
                    }
                }};
            }
            match (add.len(), sub.len()) {
                (1, 1) => pasada!(+ col(add[0]), - col(sub[0])),
                (1, 2) => pasada!(+ col(add[0]), - col(sub[0]), - col(sub[1])),
                (2, 2) => pasada!(+ col(add[0]), + col(add[1]), - col(sub[0]), - col(sub[1])),
                _ => {
                    for t in (0..h).step_by(32) {
                        let mut a = _mm256_loadu_si256(src.add(t).cast());
                        let mut b = _mm256_loadu_si256(src.add(t + 16).cast());
                        for &f in add {
                            let w = col(f).add(t);
                            a = _mm256_add_epi16(a, _mm256_loadu_si256(w.cast()));
                            b = _mm256_add_epi16(b, _mm256_loadu_si256(w.add(16).cast()));
                        }
                        for &f in sub {
                            let w = col(f).add(t);
                            a = _mm256_sub_epi16(a, _mm256_loadu_si256(w.cast()));
                            b = _mm256_sub_epi16(b, _mm256_loadu_si256(w.add(16).cast()));
                        }
                        _mm256_storeu_si256(dst.add(t).cast(), a);
                        _mm256_storeu_si256(dst.add(t + 16).cast(), b);
                    }
                }
            }
        }
    }

    pub(super) fn aplicar_features_desde_avx2(
        &self,
        src: &[i16; H_MAX],
        dst: &mut [i16; H_MAX],
        add: &[u16],
        sub: &[u16],
    ) {
        // SAFETY: AVX2 habilitado al compilar; h multiplo de 32, features
        // validadas por el llamador. Los bloques estan dentro de los arrays.
        unsafe { self.actualizar_avx2(src.as_ptr(), dst.as_mut_ptr(), add, sub) }
    }

    pub(super) fn aplicar_features_avx2(&self, acc: &mut [i16; H_MAX], add: &[u16], sub: &[u16]) {
        let p = acc.as_mut_ptr();
        unsafe { self.actualizar_avx2(p, p, add, sub) }
    }

    /// Mismas precondiciones que el kernel NEON: h<=1024, h%32==0,
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
            let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            macro_rules! combinar {
                (+ $a:expr, $b:expr) => {
                    _mm256_add_epi16($a, $b)
                };
                (- $a:expr, $b:expr) => {
                    _mm256_sub_epi16($a, $b)
                };
            }
            macro_rules! bloque {
                ($t:expr, $s:expr, $d:expr, $w:expr, $a:ident, [$($signo:tt $col:expr),+]) => {{
                    let mut x = _mm256_loadu_si256($s.add($t).cast());
                    let mut y = _mm256_loadu_si256($s.add($t + 16).cast());
                    $(
                        let w = $col.add($t);
                        x = combinar!($signo x, _mm256_loadu_si256(w.cast()));
                        y = combinar!($signo y, _mm256_loadu_si256(w.add(16).cast()));
                    )+
                    _mm256_storeu_si256($d.add($t).cast(), x);
                    _mm256_storeu_si256($d.add($t + 16).cast(), y);
                    $a = _mm256_add_epi32($a, screlu(x, $w.add($t)));
                    $a = _mm256_add_epi32($a, screlu(y, $w.add($t + 16)));
                }};
            }
            macro_rules! pasada {
                ([$($s0:tt $c0:expr),+], [$($s1:tt $c1:expr),+]) => {{
                    for t in (0..h).step_by(32) {
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
            // Por perspectiva/carril: 1024/8 terminos de como mucho
            // 65025*128 -> 1,065,369,600 < i32::MAX. Reunir en i64.
            Some(suma_i64(a0) + suma_i64(a1))
        }
    }
}

impl AcumBullet {
    #[inline(always)]
    pub(super) fn producto_punto_avx2(
        &self,
        yo: &[i16; H_MAX],
        rival: &[i16; H_MAX],
        bucket: usize,
    ) -> i64 {
        debug_assert!(self.red.l1w_cortos);
        unsafe {
            let w = self.red.l1w.as_ptr().add(bucket * 2 * H_MAX);
            let (mut a, mut b) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            for j in (0..self.red.h).step_by(16) {
                a = _mm256_add_epi32(
                    a,
                    screlu(_mm256_loadu_si256(yo.as_ptr().add(j).cast()), w.add(j)),
                );
                b = _mm256_add_epi32(
                    b,
                    screlu(
                        _mm256_loadu_si256(rival.as_ptr().add(j).cast()),
                        w.add(H_MAX + j),
                    ),
                );
            }
            suma_i64(a) + suma_i64(b)
        }
    }
}
