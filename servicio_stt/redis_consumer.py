"""
servicio_stt/redis_consumer.py
================================
Consumer de Redis Streams para el servicio STT.

Patrón de consumo: Consumer Group con ACK manual.
  - Permite múltiples instancias del servicio STT en paralelo (escalabilidad horizontal).
  - Si el proceso muere con un mensaje en vuelo, otro consumer lo reclama tras CLAIM_IDLE_MS.
  - Solo se hace ACK después de publicar exitosamente el texto al stream de salida.

Flujo por mensaje:
  1. XREADGROUP desde "agrovoz:audio"
  2. Deserializar AudioChunk → decodificar base64 → numpy float32
  3. Transcribir con WhisperEngine
  4. Serializar TranscripcionTexto → XADD a "agrovoz:texto"
  5. XACK del mensaje original (solo si el paso 4 fue exitoso)
  6. Si transcripción vacía → XACK y descartar (no propagar silencio)

Resiliencia:
  - Reconexión automática a Redis con backoff exponencial.
  - Mensajes pendientes (PEL) se reclaman al arrancar tras un crash.
  - Dead Letter Queue (DLQ) para mensajes que fallan repetidamente.
"""

import base64
import json
import logging
import os
import time
from typing import Optional

import numpy as np
import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from shared.redis_config import (
    BLOCK_MS,
    CLAIM_IDLE_MS,
    FIELD_AUDIO_PAYLOAD,
    FIELD_TEXT_PAYLOAD,
    FIELD_TEXT_SESSION_ID,
    GROUP_STT,
    MAX_MESSAGES_PER_POLL,
    STREAM_AUDIO,
    STREAM_DLQ,
    STREAM_MAXLEN,
    STREAM_TEXTO,
)
from shared.schemas import AudioChunk, TranscripcionTexto
from whisper_engine import WhisperEngine, TranscripcionResultado

logger = logging.getLogger(__name__)

# Número máximo de reintentos antes de enviar un mensaje al DLQ
MAX_DELIVERY_COUNT = 3


class STTConsumer:
    """
    Consumer del stream "agrovoz:audio" que produce en "agrovoz:texto".

    Gestiona el ciclo de vida del consumer group, la reclamación de mensajes
    pendientes y la integración con WhisperEngine.
    """

    def __init__(
        self,
        redis_url: str,
        whisper_engine: WhisperEngine,
        consumer_name: str,
        group_name:    str = GROUP_STT,
        stream_audio:  str = STREAM_AUDIO,
        stream_texto:  str = STREAM_TEXTO,
    ):
        self.redis_url      = redis_url
        self.engine         = whisper_engine
        self.consumer_name  = consumer_name
        self.group_name     = group_name
        self.stream_audio   = stream_audio
        self.stream_texto   = stream_texto

        self._client: Optional[redis.Redis] = None
        self._running = False

        # Métricas
        self._procesados   = 0
        self._descartados  = 0   # Utterances de silencio/ruido
        self._errores      = 0
        self._al_dlq       = 0

        logger.info(
            "STTConsumer '%s' inicializado → %s → %s",
            consumer_name, stream_audio, stream_texto,
        )

    # ----------------------------------------------------------
    # Ciclo de vida
    # ----------------------------------------------------------

    def iniciar(self) -> None:
        """Conecta a Redis, inicializa el consumer group y entra al loop."""
        logger.info("Iniciando STTConsumer '%s'…", self.consumer_name)
        self._conectar()
        self._asegurar_grupo_consumer()
        self._reclamar_mensajes_pendientes()

        self._running = True
        logger.info("✅ STTConsumer listo. Escuchando stream '%s'…", self.stream_audio)

        try:
            self._loop_consumo()
        except KeyboardInterrupt:
            logger.info("Interrupción recibida. Deteniendo consumer…")
        finally:
            self._running = False
            self._log_metricas_finales()

    def detener(self) -> None:
        """Señaliza al loop para detenerse limpiamente."""
        self._running = False

    # ----------------------------------------------------------
    # Loop principal
    # ----------------------------------------------------------

    def _loop_consumo(self) -> None:
        """
        Loop principal: XREADGROUP → procesar → XACK.
        Bloquea BLOCK_MS ms en cada poll si no hay mensajes.
        """
        ultima_metrica = time.time()

        while self._running:
            try:
                # Leer mensajes nuevos del consumer group
                respuesta = self._client.xreadgroup(
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    streams={self.stream_audio: ">"},  # ">" = solo mensajes no entregados aún
                    count=MAX_MESSAGES_PER_POLL,
                    block=BLOCK_MS,
                )

                if not respuesta:
                    # Timeout: no hay mensajes nuevos, continuar
                    continue

                for stream_name, mensajes in respuesta:
                    for msg_id, campos in mensajes:
                        self._procesar_mensaje(msg_id, campos)

            except (RedisConnectionError, RedisTimeoutError) as e:
                logger.error("Conexión Redis perdida en consumer: %s. Reconectando…", e)
                time.sleep(2)
                self._reconectar()

            except Exception as e:
                logger.exception("Error inesperado en loop de consumo: %s", e)
                time.sleep(1)

            # Log periódico de métricas (cada 120s)
            if time.time() - ultima_metrica >= 120:
                self._log_metricas_periodicas()
                self.engine.log_metricas()
                ultima_metrica = time.time()

    # ----------------------------------------------------------
    # Procesamiento de mensajes individuales
    # ----------------------------------------------------------

    def _procesar_mensaje(self, msg_id: bytes, campos: dict) -> None:
        """
        Pipeline completo para un mensaje:
        Deserializar → Decodificar audio → Transcribir → Publicar → ACK
        """
        logger.debug("Procesando mensaje %s…", msg_id)
        t0 = time.perf_counter()

        try:
            # 1. Deserializar payload AudioChunk
            chunk = self._deserializar_audio_chunk(campos)
            if chunk is None:
                # Mensaje malformado → ACK para no reencolar indefinidamente
                self._ack(msg_id)
                self._errores += 1
                return

            # 2. Decodificar audio base64 → numpy float32
            audio_float = self._decodificar_audio(chunk.audio_b64)

            # 3. Transcribir con Whisper
            resultado = self.engine.transcribir(audio_float)

            # 4. Si el audio era silencio o ruido → descartar
            if resultado.es_vacio or not resultado.texto.strip():
                logger.debug(
                    "Mensaje %s descartado: utterance vacío (ruido/silencio).", msg_id
                )
                self._ack(msg_id)
                self._descartados += 1
                return

            # 5. Verificar delivery count — posible loop de reintento
            delivery_count = self._obtener_delivery_count(msg_id)
            if delivery_count > MAX_DELIVERY_COUNT:
                self._enviar_dlq(msg_id, campos, f"delivery_count={delivery_count}")
                self._ack(msg_id)
                self._al_dlq += 1
                return

            # 6. Construir TranscripcionTexto
            transcripcion = TranscripcionTexto(
                chunk_id=chunk.chunk_id,
                session_id=chunk.session_id,
                texto=resultado.texto,
                idioma_detectado=resultado.idioma_detectado,
                confianza_stt=resultado.confianza,
            )

            # 7. Publicar al stream de texto
            exito = self._publicar_transcripcion(transcripcion)

            if exito:
                # 8. ACK solo tras publicación exitosa
                self._ack(msg_id)
                self._procesados += 1

                latencia_total = (time.perf_counter() - t0) * 1000
                logger.info(
                    "✅ [%d] STT '%s…' | Whisper=%.0fms | Total=%.0fms | session=%s",
                    self._procesados,
                    resultado.texto[:50],
                    resultado.latencia_ms,
                    latencia_total,
                    chunk.session_id,
                )
            else:
                # No hacer ACK: el mensaje quedará en PEL para reintento
                logger.warning(
                    "No se pudo publicar transcripción para msg_id=%s. "
                    "Mensaje quedará en PEL para reintento.", msg_id
                )
                self._errores += 1

        except Exception as e:
            logger.exception("Error procesando mensaje %s: %s", msg_id, e)
            self._errores += 1
            # No hacer ACK: el mensaje quedará en PEL

    def _deserializar_audio_chunk(self, campos: dict) -> Optional[AudioChunk]:
        """Deserializa el campo JSON del mensaje Redis en un AudioChunk Pydantic."""
        try:
            payload_json = campos.get(FIELD_AUDIO_PAYLOAD)
            if not payload_json:
                logger.error("Mensaje sin campo '%s'. Campos: %s", FIELD_AUDIO_PAYLOAD, list(campos.keys()))
                return None
            return AudioChunk.model_validate_json(payload_json)
        except Exception as e:
            logger.error("Error deserializando AudioChunk: %s", e)
            return None

    def _decodificar_audio(self, audio_b64: str) -> np.ndarray:
        """Decodifica audio base64 → int16 bytes → float32 numpy [-1, 1]."""
        audio_bytes = base64.b64decode(audio_b64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def _publicar_transcripcion(self, transcripcion: TranscripcionTexto) -> bool:
        """Publica el resultado STT al stream 'agrovoz:texto'."""
        try:
            self._client.xadd(
                name=self.stream_texto,
                fields={
                    FIELD_TEXT_PAYLOAD:    transcripcion.model_dump_json(),
                    FIELD_TEXT_SESSION_ID: transcripcion.session_id,
                },
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            return True
        except Exception as e:
            logger.error("Error publicando transcripción en '%s': %s", self.stream_texto, e)
            return False

    def _ack(self, msg_id: bytes) -> None:
        """Hace ACK del mensaje en el consumer group."""
        try:
            self._client.xack(self.stream_audio, self.group_name, msg_id)
        except Exception as e:
            logger.warning("Error en XACK para msg_id=%s: %s", msg_id, e)

    def _enviar_dlq(self, msg_id: bytes, campos: dict, razon: str) -> None:
        """Envía mensaje problemático al Dead Letter Queue para auditoría."""
        try:
            self._client.xadd(
                name=STREAM_DLQ,
                fields={
                    "msg_id_original": str(msg_id),
                    "stream_origen":   self.stream_audio,
                    "razon":           razon,
                    "consumer":        self.consumer_name,
                    "payload":         json.dumps(campos),
                },
                maxlen=1000,
            )
            logger.warning(
                "Mensaje %s enviado al DLQ. Razón: %s", msg_id, razon
            )
        except Exception as e:
            logger.error("Error enviando al DLQ: %s", e)

    def _obtener_delivery_count(self, msg_id: bytes) -> int:
        """Consulta cuántas veces fue entregado un mensaje (para detectar poison pills)."""
        try:
            pel = self._client.xpending_range(
                self.stream_audio, self.group_name,
                min=msg_id, max=msg_id, count=1,
            )
            if pel:
                return pel[0].get("times_delivered", 1)
        except Exception:
            pass
        return 1

    # ----------------------------------------------------------
    # Gestión del Consumer Group
    # ----------------------------------------------------------

    def _asegurar_grupo_consumer(self) -> None:
        """
        Crea el consumer group si no existe.
        MKSTREAM=True crea el stream si tampoco existe.
        """
        try:
            self._client.xgroup_create(
                name=self.stream_audio,
                groupname=self.group_name,
                id="0",          # Procesar desde el inicio del stream
                mkstream=True,   # Crear stream si no existe
            )
            logger.info(
                "Consumer group '%s' creado en stream '%s'.",
                self.group_name, self.stream_audio,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # El grupo ya existe — normal en reinicios del servicio
                logger.info(
                    "Consumer group '%s' ya existe en '%s'. Continuando.",
                    self.group_name, self.stream_audio,
                )
            else:
                raise

    def _reclamar_mensajes_pendientes(self) -> None:
        """
        Al arrancar, reclama mensajes que quedaron en vuelo (PEL) por un crash previo.
        Estos mensajes están en el grupo pero ningún consumer los está procesando.
        """
        try:
            pendientes = self._client.xpending(self.stream_audio, self.group_name)
            total_pendientes = pendientes.get("pending", 0)

            if total_pendientes == 0:
                logger.info("No hay mensajes pendientes en PEL.")
                return

            logger.warning(
                "Encontrados %d mensaje(s) pendiente(s) en PEL. Reclamando…",
                total_pendientes,
            )

            # Reclamar mensajes que llevan más de CLAIM_IDLE_MS sin ser procesados
            reclamados = self._client.xautoclaim(
                name=self.stream_audio,
                groupname=self.group_name,
                consumername=self.consumer_name,
                min_idle_time=CLAIM_IDLE_MS,
                start_id="0-0",
                count=100,
            )

            mensajes_reclamados = reclamados[1] if reclamados else []
            if mensajes_reclamados:
                logger.info(
                    "Reclamados %d mensajes del PEL para reprocesar.",
                    len(mensajes_reclamados),
                )
                for msg_id, campos in mensajes_reclamados:
                    self._procesar_mensaje(msg_id, campos)
        except Exception as e:
            logger.warning("Error al reclamar mensajes pendientes: %s", e)

    # ----------------------------------------------------------
    # Conexión Redis
    # ----------------------------------------------------------

    def _conectar(self) -> None:
        """Conecta a Redis con reintentos exponenciales."""
        delay = 1.0
        intento = 0
        while True:
            intento += 1
            try:
                client = redis.from_url(
                    self.redis_url,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                    retry_on_timeout=True,
                    health_check_interval=30,
                    decode_responses=True,
                )
                client.ping()
                self._client = client
                logger.info("✅ STTConsumer conectado a Redis (intento %d).", intento)
                return
            except Exception as e:
                logger.warning(
                    "Intento %d de conexión Redis fallido: %s. Reintentando en %.0fs…",
                    intento, e, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

    def _reconectar(self) -> None:
        """Reconexión tras pérdida de conexión durante el loop."""
        self._client = None
        self._conectar()
        self._asegurar_grupo_consumer()

    # ----------------------------------------------------------
    # Métricas
    # ----------------------------------------------------------

    def _log_metricas_periodicas(self) -> None:
        logger.info(
            "📊 STT Consumer '%s' — procesados: %d | descartados: %d | "
            "errores: %d | al_dlq: %d",
            self.consumer_name,
            self._procesados,
            self._descartados,
            self._errores,
            self._al_dlq,
        )

    def _log_metricas_finales(self) -> None:
        logger.info(
            "🛑 STT Consumer detenido. Procesados: %d | Descartados: %d | "
            "Errores: %d | Al DLQ: %d",
            self._procesados, self._descartados,
            self._errores, self._al_dlq,
        )
