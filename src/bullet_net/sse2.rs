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
    /// Actualizacion vectorizada SSE2 en una sola pasada (src y dst pueden coincidir).
    #[inline(always)]
    unsafe fn actualizar_sse2(&self, src: *const i16, dst: *mut i16, add: &[u16], sub: &[u16]) {
        unsafe {
            let h = self.h;
            let col = |f: u16| self.l0w.as_ptr().add(f as usize * h);
            macro_rules! combinar {
                (+ $a:expr, $b:expr) => {
                    _mm_add_epi16($a, $b)
                };
                (- $a:expr, $b:expr) => {
                    _mm_sub_epi16($a, $b)
                };
            }
            macro_rules! pasada {
                ($($signo:tt $col:expr),+ $(,)?) => {{
                    for t in (0..h).step_by(32) {
                        let mut r0 = _mm_loadu_si128(src.add(t).cast());
                        let mut r1 = _mm_loadu_si128(src.add(t + 8).cast());
                        let mut r2 = _mm_loadu_si128(src.add(t + 16).cast());
                        let mut r3 = _mm_loadu_si128(src.add(t + 24).cast());
                        $(
                            let w = $col.add(t);
                            r0 = combinar!($signo r0, _mm_loadu_si128(w.cast()));
                            r1 = combinar!($signo r1, _mm_loadu_si128(w.add(8).cast()));
                            r2 = combinar!($signo r2, _mm_loadu_si128(w.add(16).cast()));
                            r3 = combinar!($signo r3, _mm_loadu_si128(w.add(24).cast()));
                        )+
                        _mm_storeu_si128(dst.add(t).cast(), r0);
                        _mm_storeu_si128(dst.add(t + 8).cast(), r1);
                        _mm_storeu_si128(dst.add(t + 16).cast(), r2);
                        _mm_storeu_si128(dst.add(t + 24).cast(), r3);
                    }
                }};
            }
            match (add.len(), sub.len()) {
                (1, 1) => pasada!(+ col(add[0]), - col(sub[0])),
                (1, 2) => pasada!(+ col(add[0]), - col(sub[0]), - col(sub[1])),
                (2, 2) => pasada!(+ col(add[0]), + col(add[1]), - col(sub[0]), - col(sub[1])),
                _ => {
                    for t in (0..h).step_by(32) {
                        let mut r0 = _mm_loadu_si128(src.add(t).cast());
                        let mut r1 = _mm_loadu_si128(src.add(t + 8).cast());
                        let mut r2 = _mm_loadu_si128(src.add(t + 16).cast());
                        let mut r3 = _mm_loadu_si128(src.add(t + 24).cast());
                        for &f in add {
                            let w = col(f).add(t);
                            r0 = _mm_add_epi16(r0, _mm_loadu_si128(w.cast()));
                            r1 = _mm_add_epi16(r1, _mm_loadu_si128(w.add(8).cast()));
                            r2 = _mm_add_epi16(r2, _mm_loadu_si128(w.add(16).cast()));
                            r3 = _mm_add_epi16(r3, _mm_loadu_si128(w.add(24).cast()));
                        }
                        for &f in sub {
                            let w = col(f).add(t);
                            r0 = _mm_sub_epi16(r0, _mm_loadu_si128(w.cast()));
                            r1 = _mm_sub_epi16(r1, _mm_loadu_si128(w.add(8).cast()));
                            r2 = _mm_sub_epi16(r2, _mm_loadu_si128(w.add(16).cast()));
                            r3 = _mm_sub_epi16(r3, _mm_loadu_si128(w.add(24).cast()));
                        }
                        _mm_storeu_si128(dst.add(t).cast(), r0);
                        _mm_storeu_si128(dst.add(t + 8).cast(), r1);
                        _mm_storeu_si128(dst.add(t + 16).cast(), r2);
                        _mm_storeu_si128(dst.add(t + 24).cast(), r3);
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub(super) unsafe fn aplicar_features_desde_sse2(
        &self,
        src: &[i16; H_MAX],
        dst: &mut [i16; H_MAX],
        add: &[u16],
        sub: &[u16],
    ) {
        unsafe { self.actualizar_sse2(src.as_ptr(), dst.as_mut_ptr(), add, sub) }
    }

    #[inline(always)]
    pub(super) unsafe fn aplicar_features_sse2(
        &self,
        acc: &mut [i16; H_MAX],
        add: &[u16],
        sub: &[u16],
    ) {
        let p = acc.as_mut_ptr();
        unsafe { self.actualizar_sse2(p, p, add, sub) }
    }

    /// SSE2 disponible en x86-64: h<=1024, h%32==0,
    /// pesos de salida en -128..=128, features validas y bucket existente.
    #[inline(always)]
    pub(super) unsafe fn actualizar_y_salida_sse2(
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
                    _mm_storeu_si128($d.add(8 + $t).cast(), y);
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
        let tiene_avx2_hw = std::arch::is_x86_feature_detected!("avx2");
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
                                    red.actualizar_y_salida_sse2(
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
                                if tiene_avx2_hw {
                                    let mut dst_avx2 = [[12345i16; H_MAX]; 2];
                                    let suma_avx2 = unsafe {
                                        red.actualizar_y_salida_avx2(
                                            &src, &mut dst_avx2, &cambios, negras, bucket,
                                        )
                                    }
                                    .expect("forma comun debe usar el kernel AVX2");
                                    assert_eq!(dst_avx2, esperado);
                                    assert_eq!(suma_avx2, referencia);
                                }
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
        assert!(unsafe { red.actualizar_y_salida_sse2(&src, &mut dst, &cambios, false, 0) }.is_none());
        assert_eq!(dst, [[23456; H_MAX]; 2]);
    }

    #[test]
    fn sse2_y_avx2_bit_exactos_contra_escalar_en_red_real() {
        let datos = include_bytes!("../../pesos_bullet_1024_buckets8.bin");
        let red = Box::leak(Box::new(RedBullet::cargar_de_bytes(datos).unwrap()));
        let tiene_avx2_hw = std::arch::is_x86_feature_detected!("avx2");
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
        ];
        let mut semilla = 0x5EED_CAFE_1234_5678u64;
        let mut verificadas = 0usize;
        for fen in fens {
            let raiz = crate::board::Board::from_fen(fen).unwrap();
            let mut ac = AcumBullet::desde_tablero(red, &raiz);
            let mut actual = raiz;
            for ply in 0..30 {
                assert_eq!(ac.evaluar().to_bits(), ac.evaluar_escalar().to_bits());
                let persp = ac.persp();
                let bucket = ac.bucket(ac.piezas());
                let (yo, rival) = if ac.negras_mueven() {
                    (&persp[1], &persp[0])
                } else {
                    (&persp[0], &persp[1])
                };
                let esc = ac.producto_punto_escalar(yo, rival, bucket);
                let sse2 = ac.producto_punto_sse2(yo, rival, bucket);
                assert_eq!(sse2, esc, "producto_punto_sse2 difiere en ply={ply}");
                if tiene_avx2_hw {
                    let avx2 = unsafe { ac.producto_punto_avx2(yo, rival, bucket) };
                    assert_eq!(avx2, esc, "producto_punto_avx2 difiere en ply={ply}");
                }
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
                let (c0, c1, _, _) = AcumBullet::calcular_cambios(&actual, &hijo);
                let cambios = [c0, c1];
                let mut dst_sse2 = [[0i16; H_MAX]; 2];
                for p in 0..2 {
                    unsafe {
                        red.aplicar_features_desde_sse2(
                            &persp[p],
                            &mut dst_sse2[p],
                            &cambios[p].add,
                            &cambios[p].sub,
                        );
                    }
                }
                ac.entrar(&actual, &hijo);
                assert_eq!(dst_sse2, ac.persp(), "aplicar_features_desde_sse2 difiere en ply={ply}");
                actual = hijo;
            }
        }
        assert!(verificadas >= 120);
    }
}
