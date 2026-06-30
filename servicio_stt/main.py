"""
servicio_stt/main.py
======================
Entry point del servicio de transcripción de voz (STT) de AgroVoz.

Orquesta:
  1. WhisperEngine   — carga modelo Whisper (CTranslate2) en memoria
  2. STTConsumer     — consumer group Redis: audio → texto

Variables de entorno requeridas:
    REDIS_URL, REDIS_STREAM_AUDIO, REDIS_STREAM_TEXTO, REDIS_GROUP_STT,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

Escalabilidad horizontal:
    Levantar múltiples instancias con CONSUMER_NAME distinto para
    paralelizar transcripciones. El consumer group reparte los mensajes.

    STT_CONSUMER_NAME=stt_worker_1 python main.py
    STT_CONSUMER_NAME=stt_worker_2 python main.py
"""

import logging
import os
import signal
import socket
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper_engine import WhisperConfig, WhisperEngine
from redis_consumer import STTConsumer

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agrovoz.servicio_stt")


def main() -> None:
    logger.info("=" * 60)
    logger.info("🎙️  AgroVoz — Servicio STT (Whisper + CTranslate2)")
    logger.info("=" * 60)

    # Nombre único del consumer (hostname + pid para múltiples instancias)
    consumer_name = os.getenv(
        "STT_CONSUMER_NAME",
        f"stt_{socket.gethostname()}_{os.getpid()}"
    )
    logger.info("Consumer name: %s", consumer_name)

    # Configurar Whisper
    whisper_config = WhisperConfig(
        model_name   = os.getenv("WHISPER_MODEL",        "small"),
        device       = os.getenv("WHISPER_DEVICE",       "cpu"),
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        download_root= os.getenv("WHISPER_CACHE_DIR",    "/models/whisper"),
    )

    engine   = WhisperEngine(whisper_config)
    consumer = STTConsumer(
        redis_url      = os.getenv("REDIS_URL",           "redis://localhost:6379/0"),
        whisper_engine = engine,
        consumer_name  = consumer_name,
        group_name     = os.getenv("REDIS_GROUP_STT",    "grupo_stt"),
        stream_audio   = os.getenv("REDIS_STREAM_AUDIO", "agrovoz:audio"),
        stream_texto   = os.getenv("REDIS_STREAM_TEXTO", "agrovoz:texto"),
    )

    # Shutdown limpio ante SIGTERM (Podman stop)
    def handle_shutdown(sig, frame):
        logger.info("Señal %s recibida. Deteniendo servicio STT…", signal.Signals(sig).name)
        consumer.detener()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT,  handle_shutdown)

    # Paso 1: Cargar modelo Whisper en memoria
    logger.info("Cargando modelo Whisper '%s'…", whisper_config.model_name)
    engine.cargar_modelo()

    # Paso 2: Iniciar consumer (bloqueante hasta SIGTERM)
    consumer.iniciar()

    logger.info("🛑 Servicio STT finalizado.")


if __name__ == "__main__":
    main()
