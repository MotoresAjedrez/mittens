// Binario UCI de Mittens.
//
// Deliberadamente vacio: TODA la logica (uci_loop, subcomandos de linea de
// comandos, parsers, etc.) vive en la libreria `mimotor_core` (src/lib.rs),
// que este binario enlaza como rlib. Antes este archivo era una copia
// literal de src/lib.rs, y las dos copias ya habian divergido en silencio
// -- cada fix habia que aplicarlo dos veces y una de ellas se olvidaba.
//
// Si hace falta agregar comportamiento, va en src/lib.rs.
fn main() {
    mimotor_core::run_cli();
}
