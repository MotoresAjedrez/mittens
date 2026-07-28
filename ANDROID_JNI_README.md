# MiMotor Tal en Android — puente JNI (entrega para el agente de la app Kotlin)

Este documento explica cómo enchufar el motor de ajedrez (Rust) a una app
Android. **Toda la parte nativa ya está compilada y probada**; del lado
Kotlin no hace falta NDK, ni Cargo, ni ningún paso de compilación nativa en
Gradle: solo copiar los `.so` y declarar la clase.

---

## 1. La clase Kotlin que tenés que declarar (exacta)

El paquete y el nombre de la clase son **parte del nombre del símbolo** en
la librería nativa. Si los cambiás, Java tira `UnsatisfiedLinkError`.

```kotlin
package com.tavito.mimotor

object MimotorNative {
    init { System.loadLibrary("mimotor_core") }   // sin "lib" y sin ".so"

    external fun nativeNew(): Long
    external fun nativeEnviar(handle: Long, comando: String)
    external fun nativeLeerLinea(handle: Long): String?
    external fun nativeLeerLineaEsperando(handle: Long, timeoutMs: Long): String?
    external fun nativeLiberar(handle: Long)
}
```

Los métodos son **estáticos** del lado Rust (reciben `JClass`, no `JObject`),
por eso `object` en Kotlin (o `companion object` / métodos `@JvmStatic`) es lo
correcto. Si preferís una `class` normal con métodos de instancia, también
funciona: JNI pasa `this` en vez de la clase y el puente lo ignora igual.

### Símbolos exportados en el `.so`

```
Java_com_tavito_mimotor_MimotorNative_nativeNew
Java_com_tavito_mimotor_MimotorNative_nativeEnviar
Java_com_tavito_mimotor_MimotorNative_nativeLeerLinea
Java_com_tavito_mimotor_MimotorNative_nativeLeerLineaEsperando
Java_com_tavito_mimotor_MimotorNative_nativeLiberar
```

Firmas JNI equivalentes:

| Método | Firma JNI |
|---|---|
| `nativeNew` | `()J` |
| `nativeEnviar` | `(JLjava/lang/String;)V` |
| `nativeLeerLinea` | `(J)Ljava/lang/String;` |
| `nativeLeerLineaEsperando` | `(JJ)Ljava/lang/String;` |
| `nativeLiberar` | `(J)V` |

Código fuente del puente: `src/jni_bridge.rs`.

---

## 2. Dónde poner los `.so`

Ya están compilados en este repo, en:

```
target/aarch64-linux-android/release/libmimotor_core.so     (ARM 64 bits)
target/armv7-linux-androideabi/release/libmimotor_core.so   (ARM 32 bits)
target/x86_64-linux-android/release/libmimotor_core.so      (emulador x86_64)
```

Copiálos al proyecto Android con la estructura estándar de `jniLibs`
(Gradle los empaqueta solo, sin configuración extra):

```
app/src/main/jniLibs/arm64-v8a/libmimotor_core.so
app/src/main/jniLibs/armeabi-v7a/libmimotor_core.so
app/src/main/jniLibs/x86_64/libmimotor_core.so
```

**Atajo:** ya quedaron copiados con esa misma estructura en
`android-jnilibs/` de este repo, así que alcanza con:

```bash
cp -R /Users/Tavito/mi-motor-rust-produccion/android-jnilibs/* \
      /ruta/a/tu/proyecto/app/src/main/jniLibs/
```

(Si recompilás el motor, volvé a copiar desde `target/…` a
`android-jnilibs/` para que no queden desactualizados.)

`minSdk` mínimo: **24** (Android 7.0). Está horneado en el nombre del wrapper
del NDK que se usó para enlazar; si el proyecto pide un minSdk menor, hay que
recompilar con otro wrapper.

Para recompilar los `.so` después de tocar el motor:

```bash
cd /Users/Tavito/mi-motor-rust-produccion
cargo build --release --target aarch64-linux-android   --lib
cargo build --release --target armv7-linux-androideabi --lib
cargo build --release --target x86_64-linux-android    --lib
```

Los linkers del NDK ya están configurados en `.cargo/config.toml`.

---

## 3. Los pesos NNUE (`pesos_amenazas_prueba.bin`, 5.5 MB)

El motor los carga **por ruta de archivo**, con código nativo (`fopen`). El
código nativo **no puede leer directo desde adentro del APK**, así que no
alcanza con poner el archivo en `assets/` — hay que copiarlo una vez a
almacenamiento interno y pasarle esa ruta al motor.

1. Poner el archivo en `app/src/main/assets/pesos_amenazas_prueba.bin`.
2. Evitar que Gradle lo comprima (si no, la copia igual funciona, pero esto
   es más rápido y prolijo). En `app/build.gradle.kts`:

   ```kotlin
   android {
       androidResources { noCompress += "bin" }
   }
   ```

3. Copiarlo a `filesDir` en el primer arranque y pasar esa ruta al motor:

   ```kotlin
   fun rutaPesos(context: Context): String {
       val destino = File(context.filesDir, "pesos_amenazas_prueba.bin")
       if (!destino.exists() || destino.length() == 0L) {
           context.assets.open("pesos_amenazas_prueba.bin").use { entrada ->
               destino.outputStream().use { salida -> entrada.copyTo(salida) }
           }
       }
       return destino.absolutePath
   }
   ```

4. Activarlos por UCI, en este orden:

   ```kotlin
   MimotorNative.nativeEnviar(h, "setoption name NNUEPath value ${rutaPesos(ctx)}")
   MimotorNative.nativeEnviar(h, "setoption name UseNNUE value true")
   MimotorNative.nativeEnviar(h, "isready")   // esperar "readyok"
   ```

Sin NNUE el motor igual juega, pero con la evaluación clásica (bastante más
débil).

---

## 4. Cómo usarlo (flujo típico)

```kotlin
// UNA sola vez en toda la vida del proceso:
val h = MimotorNative.nativeNew()

// Handshake
MimotorNative.nativeEnviar(h, "uci")
while (true) {
    val l = MimotorNative.nativeLeerLineaEsperando(h, 3000) ?: break
    if (l == "uciok") break
}

// Pedir jugada
MimotorNative.nativeEnviar(h, "position startpos moves e2e4 e7e5")
MimotorNative.nativeEnviar(h, "go movetime 1500")
while (true) {
    val l = MimotorNative.nativeLeerLineaEsperando(h, 10_000) ?: break
    if (l.startsWith("bestmove")) { /* l.split(" ")[1] */ break }
}
```

Opciones UCI disponibles (las mismas de siempre): `Hash`, `Clear Hash`,
`Move Overhead`, `Threads`, `Personalidad` (`tal` / `universal`),
`SyzygyPath`, `BookPath`, `OwnBook`, `UseNNUE`, `NNUEPath`, `UseNN`,
`NNPath`, `QSearchNNUE`, `NNUEClassicalDepth`, `SyncUltraBullet`.

---

## 5. Advertencias importantes (leer antes de integrar)

1. **`nativeNew()` una sola vez por proceso.** Internamente el puente
   redirige los file descriptors 0 y 1 (stdin/stdout) del proceso hacia
   pipes propios y corre el `uci_loop()` original sin tocarlo. Llamarlo dos
   veces deja el proceso en un estado inconsistente. Guardá el handle en un
   singleton / `object` de Kotlin.

2. **El motor se queda con el stdout del proceso.** Después de `nativeNew()`,
   cualquier `System.out.println` o `print()` de Kotlin se va al pipe del
   motor y se pierde (o peor, aparece mezclado en las líneas UCI). Usá
   `android.util.Log` (va por logd, no por fd 1) para depurar. En una app
   normal esto es inofensivo, pero conviene saberlo.

3. **`nativeLeerLineaEsperando` bloquea.** Nunca la llames en el hilo de UI.
   Usá `Dispatchers.IO` o un hilo dedicado, y mandá los resultados a la UI
   por un `Flow` / `LiveData`.

4. **`nativeLeerLinea` no bloquea** y devuelve `null` cuando todavía no hay
   nada nuevo — sirve para pollear sin trabar, pero para el uso normal
   conviene la versión con timeout.

5. **`nativeLiberar`** cierra el canal de entrada, el `uci_loop()` ve EOF y
   termina. Después de eso el handle no sirve más y no se puede crear otro
   motor en el mismo proceso (ver punto 1). En la práctica: casi nunca la
   vas a necesitar; dejá el motor vivo mientras viva la app.

6. **`stop` funciona** (probado en el celular): mandá `"stop"` para cortar
   un `go infinite` o un `go movetime` largo; el motor responde con
   `bestmove` igual.

7. **Sin `+dotprod`.** Los `.so` de ARM se compilaron sin la instrucción
   NEON `+dotprod` que sí usa la build de la Mac (~5% más rápido en NNUE).
   Se dejó afuera a propósito: no todos los celulares Android la soportan y
   usarla sin verificar sería un `SIGILL` en tiempo de ejecución. Si algún
   día se quiere esa ganancia, hay que detectar la CPU en runtime y cargar
   un `.so` distinto.

---

## 6. Qué se verificó realmente

| Cosa | Estado |
|---|---|
| Compila `cdylib` para `aarch64-linux-android` | OK |
| Compila `cdylib` para `armv7-linux-androideabi` | OK |
| Compila `cdylib` para `x86_64-linux-android` | OK |
| Símbolos `Java_com_tavito_mimotor_MimotorNative_*` exportados en las 3 ABIs (`llvm-nm -D`) | OK |
| Arquitectura correcta de cada `.so` (`file`) | OK |
| La build de iOS (`staticlib`) sigue funcionando (`aarch64-apple-ios` y `-ios-sim`) | OK |
| Carga del `.so` y resolución JNI dentro de ART, en un **celular real** (NX809J, Android 16, arm64-v8a) | OK |
| `nativeNew()` devuelve handle válido | OK |
| Handshake `uci` → lista de opciones → `uciok` | OK |
| `isready` → `readyok` | OK |
| `nativeLeerLinea` (no bloqueante) devuelve `null` cuando no hay nada | OK |
| Carga de la NNUE real de 5.5 MB (`info string NNUE cargada ... checksum 2b890e5aa11077fb`) | OK |
| `position startpos` + `go movetime 1500` → `bestmove e2e4`, llegando a profundidad 12 (~178 k nodos) | OK |
| `position startpos moves ...` + segunda búsqueda → `bestmove` válido | OK |
| `setoption name Threads value 4` + `go infinite` + `stop` → `bestmove` | OK |
| Corrida completa repetida 3 veces, siempre "TODO OK" | OK |
| El `.so` solo depende de `libc.so`, `libm.so`, `libdl.so` (nada raro) | OK |

La prueba de humo está en `prueba-android/` y se corre con
`./prueba-android/correr-prueba.sh` con un celular o emulador conectado por
`adb`. Compila `MimotorNative.java` + `Prueba.java`, los dexea con `d8`,
empuja el `.so` y los pesos a `/data/local/tmp/mimotor/` y ejecuta la prueba
con `app_process` — o sea, dentro del runtime ART de verdad, con la
resolución de símbolos JNI real, no un `dlopen` a mano.

### Lo que NO se probó

- **La ABI `x86_64` y la `armeabi-v7a` solo se verificaron a nivel de
  compilación y de símbolos exportados**, no ejecutándolas: el único
  dispositivo disponible era arm64-v8a. Como el código es el mismo y el
  puente no tiene nada específico de arquitectura, el riesgo es bajo, pero
  conviene que el emulador x86_64 se pruebe cuando exista la app.
- **No se probó dentro de un APK con Activity/Compose** — eso es
  justamente lo que sigue del lado Kotlin. Lo que sí quedó probado es la
  parte difícil: que ART encuentra y llama los símbolos JNI por nombre.
- **Tiempos**: corriendo por `app_process` desde `adb shell` (proceso de
  shell, sin prioridad de primer plano, Android lo manda a los núcleos
  chicos), un `go movetime 1500` tardó entre 3 y 16 segundos de reloj en
  devolver `bestmove`, aunque el motor internamente reportaba `time 1091`
  en profundidad 12. Dentro de una app en primer plano esto debería
  comportarse mucho mejor, pero **conviene medirlo de nuevo** cuando la app
  exista, y si el overshoot persiste, subir `Move Overhead`.

## 7. Herramientas instaladas en esta Mac para todo esto

- **SDK de Android en `~/Library/Android/sdk`**, con las licencias ya
  aceptadas. Instalado y verificado: `platform-tools` (r37), `build-tools;34.0.0`
  y `platforms;android-34`. Poné en tu shell:

  ```bash
  export ANDROID_SDK_ROOT=$HOME/Library/Android/sdk
  export ANDROID_HOME=$ANDROID_SDK_ROOT
  export PATH=$ANDROID_SDK_ROOT/platform-tools:$PATH
  ```

  El paquete `emulator` y la imagen `system-images;android-34;google_apis;arm64-v8a`
  se dejaron descargando aparte (son ~2 GB y la conexión estaba lenta). Si no
  aparecieron todavía, terminá con:

  ```bash
  sdkmanager --sdk_root=$ANDROID_SDK_ROOT "emulator" "system-images;android-34;google_apis;arm64-v8a"
  ```

  Después, `./prueba-android/arrancar-emulador.sh` crea el AVD `mimotor34`
  (arm64-v8a, nativo en Apple Silicon, sin traducción) y lo arranca.

- **NDK 29** (ya estaba) en
  `/opt/homebrew/Caskroom/android-ndk/29/AndroidNDK14206865.app/Contents/NDK`.
  Solo hace falta si querés recompilar el motor; para la app Kotlin, no.

- **JDK**: hay OpenJDK 26 en `/opt/homebrew/opt/openjdk`, que alcanza para
  `sdkmanager`, `javac` y `d8`, pero **Gradle/AGP no soportan el 26**. Para
  el proyecto Android usá JDK 17 o 21 (`brew install openjdk@17`, después
  `export JAVA_HOME=/opt/homebrew/opt/openjdk@17`). Se dejó instalando en
  segundo plano junto con `gradle`; verificá con
  `ls /opt/homebrew/opt | grep openjdk`.

- **`adb`**: hay un celular real emparejado por Wi-Fi (`192.168.0.21:5555`,
  NX809J / Android 16 / arm64-v8a). Es el que se usó para las pruebas.
