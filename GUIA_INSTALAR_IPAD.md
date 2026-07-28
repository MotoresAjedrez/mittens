# Guía: instalar MiMotorTal en tu iPhone o iPad

Esta guía es para instalar la app **MiMotorTal** (tu motor de ajedrez) en un
iPhone o iPad **tuyo**, usando solo un **Apple ID gratuito**. No hace falta
pagar el programa de desarrollador de Apple (los 99 dólares al año). Apple
permite este camino gratis; se llama *firmar con un "Personal Team"*.

No hace falta saber programar. Solo hay que seguir los pasos en orden.

> **Lo único que no puedo hacer yo por ti**: escribir tu Apple ID y tu
> contraseña en Xcode. Eso lo tienes que hacer tú con tus propias manos, en el
> paso 3. Todo lo demás son clics.

---

## Antes de empezar: lo que ya está listo

Estas piezas ya están construidas y verificadas, no tienes que tocarlas:

| Pieza | Dónde está | Qué es |
|---|---|---|
| Motor compilado | `~/mi-motor-rust-produccion/MimotorCore.xcframework` | El motor de ajedrez, ya compilado para iPhone/iPad reales **y** para el simulador |
| Interfaz del motor | `~/mi-motor-rust-produccion/include/mimotor.h` | El "enchufe" por el que la app le habla al motor |
| Pesos de la red neuronal | `~/mi-motor-rust-produccion/pesos_amenazas_prueba.bin` | El cerebro NNUE (5.5 MB) |
| La app | `~/MiMotorTalApp/` | La aplicación en sí (tablero, botones, etc.) |

Ya se probó que el motor **funciona de verdad dentro de iOS**: arranca,
carga la red neuronal, piensa y devuelve jugadas. No es un maquillaje.

### Lo que necesitas tener a mano

- El Mac donde estamos trabajando, con **Xcode 26.6** (ya instalado).
- Tu iPhone o iPad.
- Un **cable USB** para conectarlo al Mac (la primera vez es mucho más fácil
  con cable que sin él).
- Tu **Apple ID** (el mismo correo y contraseña que usas en el App Store).
  Sirve cualquier Apple ID gratuito.

---

## Paso 1 — Generar el proyecto de Xcode

La app se describe en un archivo de recetas (`project.yml`) y de ahí se genera
el proyecto de Xcode. Abre la app **Terminal** y pega esto, dando Enter al final:

```
cd ~/MiMotorTalApp && xcodegen generate
```

Deberías ver un mensaje de que creó `MiMotorTal.xcodeproj`.

> Si te dice `command not found: xcodegen`, instálalo con:
> `brew install xcodegen` y vuelve a intentar.

**Repite este paso cada vez que se cambie algo de la estructura de la app.**
Si solo se cambió código, no hace falta.

---

## Paso 2 — Abrir el proyecto en Xcode

En la Terminal:

```
open ~/MiMotorTalApp/MiMotorTal.xcodeproj
```

Se abrirá Xcode. La primera vez puede tardar un poco mientras "indexa" los
archivos; es normal, espera a que la barrita de arriba deje de moverse.

---

## Paso 3 — Meter tu Apple ID en Xcode

Esto se hace **una sola vez** en la vida del Mac.

1. En la barra de menús: **Xcode → Settings…** (en español, *Ajustes…*).
   Atajo: `⌘ + ,`
2. Pestaña **Accounts** (*Cuentas*).
3. Botón **+** abajo a la izquierda → elige **Apple ID** → **Continue**.
4. Escribe tu correo y contraseña de Apple ID. Si tienes verificación en dos
   pasos, te llegará un código a tu iPhone; escríbelo.
5. Cuando termine, en la lista de la izquierda aparecerá tu correo, y a la
   derecha una fila que dice algo como **"Tu Nombre (Personal Team)"**.
   Eso es exactamente lo que necesitábamos.
6. Cierra la ventana de Ajustes.

---

## Paso 4 — Decirle a la app que use tu equipo personal

1. En Xcode, en la columna izquierda, haz clic en el ícono azul de arriba del
   todo que dice **MiMotorTal**.
2. En el centro aparecerán **TARGETS**; selecciona **MiMotorTal**.
3. Arriba, pestaña **Signing & Capabilities** (*Firma y capacidades*).
4. Marca la casilla **Automatically manage signing** (*Gestionar la firma
   automáticamente*) si no está marcada.
5. En el desplegable **Team** (*Equipo*), elige **"Tu Nombre (Personal Team)"**.
6. Mira el campo **Bundle Identifier**. Dirá `com.tavito.MiMotorTal`.

   > **Si aparece un error rojo** que dice que ese identificador ya está en uso
   > o que no se puede registrar: cámbialo por uno único tuyo, por ejemplo
   > `com.tavito.MiMotorTal.cesar2026`. Con eso el error desaparece. Puedes
   > poner lo que quieras mientras tenga puntos y no lleve espacios ni acentos.

### ⚠️ Un ajuste que hay que quitar para instalar en un aparato real

El proyecto viene preparado para el **simulador**, y trae desactivada la firma
de código. Para un iPhone/iPad de verdad, la firma es **obligatoria**. Si
Xcode se queja de esto, hay dos formas de arreglarlo:

**Opción A (permanente, recomendada).** Abre el archivo
`~/MiMotorTalApp/project.yml` con la app **TextEdit** y borra estas tres líneas
del final:

```
        CODE_SIGNING_REQUIRED: NO
        CODE_SIGNING_ALLOWED: NO
        CODE_SIGN_IDENTITY: ""
```

Guarda, y vuelve a hacer el **Paso 1** (`xcodegen generate`) y el **Paso 2**.

**Opción B (rápida, solo para esta vez).** En Xcode, pestaña **Build Settings**,
busca `CODE_SIGNING_ALLOWED` y ponlo en **Yes**.

---

## Paso 5 — Conectar el iPhone o iPad

1. Conéctalo al Mac con el **cable USB**.
2. En la pantalla del aparato aparecerá **"¿Confiar en este ordenador?"** →
   toca **Confiar** y escribe el código de desbloqueo del aparato.
3. La primera vez, el iPhone/iPad tiene que estar **desbloqueado** y en la
   pantalla de inicio.
4. Si es la primera vez que usas ese aparato para desarrollar, ve en el
   aparato a **Ajustes → Privacidad y seguridad → Modo de desarrollador**
   (*Developer Mode*) y actívalo. El aparato se reiniciará y te pedirá
   confirmar. Esto es normal y solo pasa una vez.
5. Vuelve a Xcode. Arriba, al centro, hay un selector que dice algo como
   `MiMotorTal > iPhone 17 Pro`. Haz clic en la parte derecha (el nombre del
   aparato) y elige **tu iPhone/iPad de verdad** de la lista. Aparecerá bajo
   un encabezado que dice *iOS Device* o con el nombre que le pusiste.

> **Sin cable (opcional).** Una vez que ya lo emparejaste con cable, en Xcode
> puedes ir a **Window → Devices and Simulators**, seleccionar tu aparato y
> marcar **"Connect via network"**. A partir de ahí, mientras estén en el mismo
> WiFi, ya no necesitas el cable.

---

## Paso 6 — Instalar la app

1. En Xcode, pulsa el botón **▶ (Run)** arriba a la izquierda. Atajo: `⌘ + R`.
2. Xcode compilará y copiará la app al aparato. Tarda un par de minutos la
   primera vez.
3. **La primera vez la app NO va a abrirse.** Verás un error en Xcode que dice
   algo como *"Could not launch... The application could not be verified"* o
   *"Untrusted Developer"*. **Esto es normal y esperado.** Falta que le digas
   a tu aparato que confía en ti.

### Confiar en tu propio certificado

En el **iPhone o iPad** (no en el Mac):

1. Abre **Ajustes**.
2. **General**.
3. Busca **VPN y gestión de dispositivos** (*VPN & Device Management*).

   > El nombre exacto de este menú **puede variar un poco** según la versión de
   > iOS. En otras versiones se llama *Gestión de dispositivos*, *Gestión de
   > perfiles y dispositivos*, o está dentro de **Ajustes → General → Perfiles**.
   > Busca cualquiera de esos nombres; siempre está dentro de *General*.

4. Verás una sección **APP DE DESARROLLADOR** con tu correo de Apple ID.
   Tócalo.
5. Toca **Confiar en "tu-correo@ejemplo.com"** y confirma tocando **Confiar**
   otra vez.

Ahora vuelve a la pantalla de inicio, busca el ícono de **MiMotorTal** y ábrelo.
Ya debería funcionar. (También puedes volver a Xcode y darle a ▶ otra vez.)

---

## ⚠️ Importante: la app caduca cada 7 días

Esta es **la limitación real** de usar un Apple ID gratuito, y conviene saberla
desde el principio para que no te agarre por sorpresa:

- La app instalada así **deja de funcionar a los ~7 días**. Al abrirla te dirá
  algo como que ya no está disponible o simplemente no abrirá.
- **No se pierde nada.** Para revivirla, solo conecta el aparato al Mac, abre
  Xcode y dale a **▶ (Run)** otra vez. Vuelve a quedar buena por otros 7 días.
- No hace falta repetir los pasos 1 al 5. Solo abrir Xcode y darle a ▶.
- Con un Apple ID gratuito también hay un tope de **3 apps propias instaladas**
  a la vez, y hasta 10 identificadores nuevos por semana. Para una sola app
  como esta, no te vas a topar con eso.

Si algún día te cansa repetir esto cada semana, la única solución es pagar el
programa de desarrollador de Apple (99 USD/año), que sube la duración a **1 año**.
Para uso personal, no hace falta.

---

## Dónde van los pesos de la red neuronal (NNUE)

El "cerebro" del motor es el archivo `pesos_amenazas_prueba.bin` (5.5 MB). El
motor **no lo lleva dentro**: lo lee de un archivo cuando arranca. Por eso tiene
que viajar dentro de la app.

**Cómo funciona el mecanismo** (verificado en el código del motor):

1. El archivo se copia a la carpeta `~/MiMotorTalApp/Resources/`.
2. `project.yml` ya tiene declarada esa carpeta como *recursos*, así que
   Xcode mete el archivo dentro de la app automáticamente al compilar.
3. Al arrancar, la app le pregunta a iOS dónde quedó el archivo y le manda al
   motor dos órdenes:
   - `setoption name NNUEPath value <la ruta del archivo>`
   - `setoption name UseNNUE value true`
4. El motor responde `info string NNUE cargada desde ... checksum ...`, y a
   partir de ahí juega con la red neuronal.

**Para copiarlo** (si aún no está), en la Terminal:

```
mkdir -p ~/MiMotorTalApp/Resources
cp ~/mi-motor-rust-produccion/pesos_amenazas_prueba.bin ~/MiMotorTalApp/Resources/
```

Y después vuelve a hacer el **Paso 1** (`xcodegen generate`).

> **Cómo saber si de verdad se cargó:** si la app tiene una zona donde muestra
> los mensajes del motor, debe aparecer la línea `NNUE cargada desde...`. Si en
> cambio aparece `error cargando NNUE...` o nunca aparece nada, es que el
> archivo no llegó a la app: revisa que esté en `Resources/` y repite el Paso 1.
>
> Esta parte depende de cómo quedó escrita la app al final. Si el que la
> programó lo hizo de otra forma (por ejemplo, descargando los pesos), esta
> sección habría que ajustarla. El mecanismo descrito es el que corresponde al
> `project.yml` tal como está hoy.

---

## Si algo sale mal

| Lo que ves | Qué hacer |
|---|---|
| `command not found: xcodegen` | `brew install xcodegen` |
| No aparece "(Personal Team)" en Team | Revisa el Paso 3; el Apple ID no quedó bien añadido |
| "Failed to register bundle identifier" | Cambia el Bundle Identifier por uno único (Paso 4) |
| "Signing requires a development team" | Elige tu Personal Team en el desplegable Team (Paso 4) |
| "code signing is not allowed" / no deja instalar en el aparato | Es el ajuste del Paso 4, sección ⚠️ |
| "Untrusted Developer" / no abre la app | Falta confiar en el certificado (Paso 6) |
| Tu iPhone no aparece en la lista de Xcode | Desbloquéalo, toca "Confiar" en el aparato, activa el Modo de desarrollador (Paso 5) |
| La app dejó de abrir después de unos días | Caducó. Conecta y dale a ▶ en Xcode |
| El motor juega pero muy flojo | Probablemente no cargó la NNUE. Mira la sección de los pesos |

---

## Resumen para la próxima vez

Una vez hecho todo esto la primera vez, revivir la app cada semana es solo:

1. Conectar el iPhone/iPad al Mac.
2. `open ~/MiMotorTalApp/MiMotorTal.xcodeproj`
3. Botón **▶**.

Nada más.
