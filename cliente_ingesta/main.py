"""
cliente_ingesta/main.py
========================
Entry point del servicio de ingesta de audio de AgroVoz.

Orquesta el pipeline de tres capas del primer tercio:
    1. AudioCapture  — captura PCM 16kHz del micrófono
    2. VADProcessor  — detecta utterances de voz con Silero VAD
    3. RedisPublisher — publica utterances al stream Redis "agrovoz:audio"

Diseño de resiliencia:
  - Si Redis cae: los utterances se guardan en buffer local y se reenvían al reconectar.
  - Si el micrófono falla: se reintenta abrir el dispositivo de audio.
  - Si el proceso recibe SIGTERM/SIGINT: shutdown limpio con flush del buffer.

Variables de entorno requeridas (ver .env.example):
    REDIS_URL, REDIS_STREAM_AUDIO,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_FRAMES,
    VAD_THRESHOLD, VAD_MIN_SILENCE_MS, VAD_MIN_SPEECH_MS, VAD_PRE_SPEECH_PAD_MS

Uso:
    python main.py
    python main.py --device-index 1  # Usar micrófono específico
    python main.py --session-id agricultor_juan_001
"""

import argparse
import logging
import os
import signal
import sys
import time
import uuid
from typing import Optional

# Asegurar que shared/ sea importable
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyaudio

from audio_capture import AudioCapture, AudioConfig
from redis_publisher import RedisPublisher, PublisherConfig
from vad_processor import VADProcessor, VADConfig

# ===========================================================
# Configuración de logging
# ===========================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agrovoz.cliente_ingesta")


# ===========================================================
# Carga de configuración desde variables de entorno
# ===========================================================

def cargar_audio_config(device_index: Optional[int] = None) -> AudioConfig:
    return AudioConfig(
        sample_rate  = int(os.getenv("AUDIO_SAMPLE_RATE",   "16000")),
        chunk_frames = int(os.getenv("AUDIO_CHUNK_FRAMES",  "8000")),
        channels     = int(os.getenv("AUDIO_CHANNELS",      "1")),
        format       = pyaudio.paInt16,
        device_index = device_index,
    )


def cargar_vad_config() -> VADConfig:
    return VADConfig(
        sample_rate       = int(os.getenv("AUDIO_SAMPLE_RATE",      "16000")),
        threshold         = float(os.getenv("VAD_THRESHOLD",         "0.5")),
        min_silence_ms    = int(os.getenv("VAD_MIN_SILENCE_MS",      "700")),
        min_speech_ms     = int(os.getenv("VAD_MIN_SPEECH_MS",       "250")),
        pre_speech_pad_ms = int(os.getenv("VAD_PRE_SPEECH_PAD_MS",  "200")),
    )


def cargar_publisher_config() -> PublisherConfig:
    return PublisherConfig(
        redis_url    = os.getenv("REDIS_URL",          "redis://localhost:6379/0"),
        stream_name  = os.getenv("REDIS_STREAM_AUDIO", "agrovoz:audio"),
    )


# ===========================================================
# Clase principal del servicio
# ===========================================================

class ClienteIngesta:
    """
    Orquestador del pipeline de ingesta de audio.
    Gestiona el ciclo de vida de todos los componentes y el shutdown limpio.
    """

    def __init__(self, session_id: str, device_index: Optional[int]):
        self.session_id  = session_id
        self._running    = False
        self._shutdown   = False

        # Instanciar componentes
        self._capture   = AudioCapture(cargar_audio_config(device_index))
        self._vad       = VADProcessor(cargar_vad_config())
        self._publisher = RedisPublisher(
            cargar_publisher_config(),
            session_id=session_id,
        )

        # Registrar handlers de señales para shutdown limpio
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT,  self._handle_shutdown)

        logger.info(
            "ClienteIngesta creado — session_id='%s', device_index=%s",
            session_id, device_index,
        )

    def iniciar(self) -> None:
        """Inicia todos los componentes y entra al bucle principal de procesamiento."""
        logger.info("=" * 60)
        logger.info("🌾 AgroVoz — Cliente de Ingesta de Audio")
        logger.info("   Session ID: %s", self.session_id)
        logger.info("=" * 60)

        try:
            # 1. Cargar modelo VAD (puede tardar en primer arranque)
            logger.info("Paso 1/3: Cargando modelo Silero VAD…")
            self._vad.cargar_modelo()

            # 2. Conectar a Redis (con reintentos si no está disponible)
            logger.info("Paso 2/3: Conectando a Redis…")
            self._publisher.conectar()

            # 3. Iniciar captura de audio del micrófono
            logger.info("Paso 3/3: Iniciando captura de audio del micrófono…")
            self._capture.start()

            self._running = True
            logger.info("✅ Pipeline de ingesta listo. Escuchando al agricultor…")
            logger.info("   (Ctrl+C para detener)")

            # Entrar al bucle principal
            self._bucle_principal()

        except KeyboardInterrupt:
            logger.info("Interrupción de teclado recibida.")
        except Exception as e:
            logger.exception("Error crítico en ClienteIngesta: %s", e)
            sys.exit(1)
        finally:
            self._detener()

    # ----------------------------------------------------------
    # Bucle principal de procesamiento
    # ----------------------------------------------------------

    def _bucle_principal(self) -> None:
        """
        Loop: lee chunks de audio → aplica VAD → publica utterances a Redis.

        Este bucle corre en el hilo principal. El audio se captura en un
        hilo daemon separado (AudioCapture).
        """
        utterances_procesados = 0
        ultima_metrica = time.time()

        while not self._shutdown:
            # Obtener chunk de audio (bloquea hasta 1s; permite chequear shutdown)
            raw_audio = self._capture.get_chunk(timeout=1.0)

            if raw_audio is None:
                # Timeout: no hay audio, continuar (permite chequear shutdown)
                continue

            # Aplicar VAD: ¿este chunk completa un utterance?
            try:
                utterance = self._vad.procesar_chunk(raw_audio)
            except Exception as e:
                logger.error("Error en VAD: %s", e)
                continue

            if utterance is not None:
                # ¡Voz detectada! Publicar a Redis.
                utterances_procesados += 1
                logger.info(
                    "[Utterance #%d] Duración: %.0fms, Confianza VAD: %.3f",
                    utterances_procesados,
                    utterance.duracion_ms,
                    utterance.confianza_prom,
                )

                self._publisher.publicar(utterance)

                # Log de estado del buffer local si hay pendientes
                if self._publisher.buffer_local_size > 0:
                    logger.warning(
                        "📥 Buffer local: %d mensaje(s) pendientes de reenvío a Redis.",
                        self._publisher.buffer_local_size,
                    )

            # Log periódico de métricas (cada 60s)
            ahora = time.time()
            if ahora - ultima_metrica >= 60:
                self._log_metricas(utterances_procesados)
                ultima_metrica = ahora

    # ----------------------------------------------------------
    # Shutdown y limpieza
    # ----------------------------------------------------------

    def _handle_shutdown(self, signum, frame) -> None:
        """Handler de señales SIGTERM/SIGINT."""
        sig_nombre = signal.Signals(signum).name
        logger.info("Señal %s recibida. Iniciando shutdown limpio…", sig_nombre)
        self._shutdown = True

    def _detener(self) -> None:
        """Detiene todos los componentes en orden inverso al arranque."""
        logger.info("Deteniendo componentes del pipeline…")

        # 1. Detener captura de audio
        if self._capture.is_running:
            self._capture.stop()

        # 2. Desconectar Redis (loggea métricas finales)
        self._publisher.desconectar()

        logger.info("🛑 Cliente de ingesta detenido correctamente.")

    def _log_metricas(self, utterances: int) -> None:
        """Log periódico del estado del sistema."""
        m = self._publisher.metricas
        logger.info(
            "📊 Métricas — utterances: %d | enviados: %d | fallidos: %d | "
            "buffer_local: %d | reconexiones: %d | cola_audio: %d",
            utterances,
            m.mensajes_enviados,
            m.mensajes_fallidos,
            self._publisher.buffer_local_size,
            m.reconexiones,
            self._capture.queue_size,
        )


# ===========================================================
# Entry point
# ===========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgroVoz — Cliente de Ingesta de Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py
  python main.py --session-id agricultor_juan
  python main.py --device-index 2
  python main.py --list-devices
        """,
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="ID de sesión del agricultor. Por defecto: UUID aleatorio.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Índice del dispositivo de micrófono. Por defecto: dispositivo del sistema.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Listar dispositivos de audio disponibles y salir.",
    )
    return parser.parse_args()


def listar_dispositivos_audio() -> None:
    """Lista todos los dispositivos de entrada de audio disponibles en el sistema."""
    p = pyaudio.PyAudio()
    print("\n📻 Dispositivos de audio de entrada disponibles:")
    print("-" * 60)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(
                f"  [{i}] {info['name']}\n"
                f"       Canales: {info['maxInputChannels']}, "
                f"Frecuencia: {int(info['defaultSampleRate'])} Hz"
            )
    print("-" * 60)
    p.terminate()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        listar_dispositivos_audio()
        sys.exit(0)

    session_id = args.session_id or f"sesion_{uuid.uuid4().hex[:8]}"

    cliente = ClienteIngesta(
        session_id=session_id,
        device_index=args.device_index,
    )
    cliente.iniciar()


if __name__ == "__main__":
    main()
