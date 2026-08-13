# Guía: cómo dejar que otra gente use Mittens SIN conectar su aparato a tu Mac

Esta guía responde a una pregunta concreta: *"¿cómo hago para que mis amigos
jueguen contra Mittens sin tener que conectar su teléfono a mi Mac?"*

La respuesta corta: **depende del aparato de tu amigo**, y para iPhone hay un
límite de Apple que ningún truco salta. Abajo están todas las opciones reales,
ordenadas de la más fácil a la más complicada.

---

## Opción 1 — Jugarlo por internet (Lichess). LA MÁS FÁCIL. Cualquier aparato, cero instalación.

Mittens puede correr como **bot de Lichess**. Cuando el bot está encendido,
**cualquier persona en el mundo**, desde el celular, tablet o PC que sea, entra
a lichess.org y lo reta a una partida. No instalan nada, no tocan tu Mac, no
importa si tienen iPhone, Android o computadora.

- **Lo que conectan es a Lichess, no a tu Mac.** Ellos solo abren una página web.
- **El único requisito**: el bot tiene que estar *encendido* para que puedan
  jugarle. El bot corre en tu Mac (ver carpeta `~/mimotor-lichess-bot`), así que
  mientras tu Mac lo tenga corriendo, tus amigos le juegan desde donde quieran.
  Si apagas el bot o la Mac, el bot aparece "offline" y no pueden retarlo (pero
  no se rompe nada, lo vuelves a encender cuando quieras).

**Esta es la que recomiendo** para "que mis amigos lo prueben sin complicaciones":
les pasas el nombre del bot en Lichess y ya. Funciona igual en iPhone que en
Android que en PC.

---

## Opción 2 — Windows: un solo archivo, sin Mac, permanente.

Para amigos con **computadora Windows**, hay un `.exe` que corre el motor sin
nada de Mac, gratis y para siempre (no caduca):

- El archivo es [`mittens-windows-x64.exe`](mittens-windows-x64.exe) (~4 MB), en
  la raíz de este repo. Está recompilado con la versión más reciente del motor
  (red NNUE nueva, modo puro, +13-16% más rápido).
- **Ojo**: es un *motor UCI*, no una app con tablero. Tu amigo lo carga en un
  programa de ajedrez gratuito para Windows como **Arena**, **BanksiaGUI** o
  **Cute Chess** (todos gratis), y ahí juega contra él con tablero y todo.
- Pasos para tu amigo: descargar el `.exe`, abrir Arena, menú *Engines →
  Install New Engine*, elegir el `.exe`, y listo. Cero conexión a tu Mac.

---

## Opción 3 — Android: compartir un APK (gratis, sin Mac, permanente) — PENDIENTE DE CONSTRUIR

En Android **sí se puede** distribuir sin ningún Mac: se genera un archivo `.apk`
y quien lo quiera lo instala directo en su teléfono (activando "instalar apps de
orígenes desconocidos"). No caduca, no necesita cable ni tu computadora.

- Lo que ya existe hoy: las librerías nativas del motor compiladas para Android
  (`android-jnilibs/`, con arm64, arm 32 bits y x86_64) y el puente para
  enchufarlas a una app Kotlin (ver [`ANDROID_JNI_README.md`](ANDROID_JNI_README.md)).
- Lo que **falta**: envolver eso en una app Android completa (el tablero, los
  botones) y exportarla como `.apk`. Eso es un proyecto de app aparte, todavía
  no está hecho. Cuando exista el `.apk`, esta es la mejor vía "sin Mac" para
  gente con Android: se los mandas por WhatsApp y lo instalan.

---

## Opción 4 — iPhone/iPad SIN conectar al Mac: solo con cuenta de pago (TestFlight)

Aquí está el límite duro de Apple, dicho sin rodeos:

- **Con Apple ID gratis NO se puede.** El método gratuito (ver
  [`GUIA_INSTALAR_IPAD.md`](GUIA_INSTALAR_IPAD.md)) OBLIGA a que cada iPhone se
  conecte a un Mac con Xcode y cable, al menos la primera vez, y encima la app
  caduca cada 7 días. Esto no es culpa de la guía ni algo que se pueda "arreglar"
  con código: es una restricción de Apple para el "Personal Team" gratuito.
- **La ÚNICA forma de que tus amigos instalen en su iPhone sin tocar tu Mac** es
  **TestFlight**, y para eso hace falta la **cuenta de desarrollador de Apple de
  pago (99 USD/año)**. Con ella:
  1. Subes la app una vez a App Store Connect.
  2. Apple te da un **link de TestFlight**.
  3. Le mandas el link a quien quieras (hasta 10.000 personas). Ellos instalan
     la app **TestFlight** gratis desde la App Store, abren tu link, y ya tienen
     Mittens — sin cable, sin Mac, sin caducar cada 7 días (dura 90 días por
     build, y subes una nueva cuando quieras).

Si de verdad quieres repartir Mittens a varios iPhones cómodamente, los 99
USD/año de Apple son el único camino. Para uno o dos aparatos tuyos, el método
gratis con cable (la otra guía) alcanza.

---

## Resumen: ¿qué le digo a cada amigo?

| Tu amigo tiene… | Qué hace | ¿Toca tu Mac? | ¿Cuesta? |
|---|---|---|---|
| Cualquier cosa (iPhone/Android/PC) | Juega el bot en **Lichess** | No | Gratis (tu Mac corre el bot) |
| **Windows** | Descarga el `.exe` + Arena | No | Gratis, permanente |
| **Android** | Instala el `.apk` *(cuando exista)* | No | Gratis, permanente |
| **iPhone**, sin tu Mac | **TestFlight** | No | 99 USD/año (Apple) |
| **iPhone**, método gratis | Cable + Xcode en tu Mac | **Sí** | Gratis, caduca 7 días |

La forma más simple para "que lo prueben ya, desde cualquier teléfono, sin
instalar nada": **el bot de Lichess** (Opción 1).
