"""
backend_api/redis_consumer.py
================================
Consumer del stream "agrovoz:texto" para el backend API.

Patrón idéntico al STTConsumer: Consumer Group con ACK manual,
reclamación de mensajes pendientes (PEL) al arrancar, DLQ para
mensajes que fallan repetidamente.

Flujo por mensaje:
  1. XREADGROUP desde "agrovoz:texto"
  2. Deserializar TranscripcionTexto
  3. Ejecutar PipelineService (Gemini + RAG + SQL)
  4. Publicar RespuestaTTS al stream "agrovoz:respuesta"
  5. XACK del mensaje original
"""

import json
import logging
import time
from typing import Optional

import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from shared.redis_config import (
    BLOCK_MS,
    CLAIM_IDLE_MS,
    FIELD_RESP_PAYLOAD,
    FIELD_RESP_SESSION_ID,
    FIELD_TEXT_PAYLOAD,
    GROUP_BACKEND,
    MAX_MESSAGES_PER_POLL,
    STREAM_DLQ,
    STREAM_MAXLEN,
    STREAM_RESPUESTA,
    STREAM_TEXTO,
)
from shared.schemas import RespuestaTTS, TranscripcionTexto
from services.pipeline_service import PipelineService

logger = logging.getLogger(__name__)

MAX_DELIVERY_COUNT = 3


class BackendConsumer:
    """
    Consumer del stream "agrovoz:texto" que produce en "agrovoz:respuesta".
    Integra el PipelineService (Gemini + RAG + SQL) de forma asíncrona.
    """

    def __init__(
        self,
        redis_url:        str,
        pipeline_service: PipelineService,
        consumer_name:    str,
        group_name:       str = GROUP_BACKEND,
        stream_texto:     str = STREAM_TEXTO,
        stream_respuesta: str = STREAM_RESPUESTA,
    ):
        self.redis_url        = redis_url
        self.pipeline         = pipeline_service
        self.consumer_name    = consumer_name
        self.group_name       = group_name
        self.stream_texto     = stream_texto
        self.stream_respuesta = stream_respuesta

        self._client:  Optional[redis.Redis] = None
        self._running  = False

        self._procesados  = 0
        self._errores     = 0
        self._al_dlq      = 0

        logger.info(
            "BackendConsumer '%s' inicializado → %s → %s",
            consumer_name, stream_texto, stream_respuesta,
        )

    # ----------------------------------------------------------
    # Ciclo de vida
    # ----------------------------------------------------------

    async def iniciar(self) -> None:
        """Conecta a Redis, inicializa el consumer group y entra al loop async."""
        logger.info("Iniciando BackendConsumer '%s'…", self.consumer_name)
        self._conectar()
        self._asegurar_grupo_consumer()
        await self._reclamar_mensajes_pendientes()

        self._running = True
        logger.info(
            "✅ BackendConsumer listo. Escuchando stream '%s'…",
            self.stream_texto,
        )

        try:
            await self._loop_consumo()
        except Exception as e:
            logger.exception("Error fatal en BackendConsumer: %s", e)
        finally:
            self._running = False
            self._log_metricas_finales()

    def detener(self) -> None:
        self._running = False

    # ----------------------------------------------------------
    # Loop principal (async)
    # ----------------------------------------------------------

    async def _loop_consumo(self) -> None:
        """
        Loop async: XREADGROUP → pipeline async → XACK.
        Usa asyncio para que las llamadas a Gemini y PostgreSQL no bloqueen.
        """
        import asyncio

        ultima_metrica = time.time()

        while self._running:
            try:
                # Redis-py es síncrono — usamos run_in_executor para no bloquear el event loop
                loop = asyncio.get_event_loop()
                respuesta = await loop.run_in_executor(
                    None,
                    lambda: self._client.xreadgroup(
                        groupname=self.group_name,
                        consumername=self.consumer_name,
                        streams={self.stream_texto: ">"},
                        count=MAX_MESSAGES_PER_POLL,
                        block=BLOCK_MS,
                    ),
                )

                if not respuesta:
                    continue

                for stream_name, mensajes in respuesta:
                    for msg_id, campos in mensajes:
                        await self._procesar_mensaje(msg_id, campos)

            except (RedisConnectionError, RedisTimeoutError) as e:
                logger.error("Conexión Redis perdida en BackendConsumer: %s. Reconectando…", e)
                await asyncio.sleep(2)
                self._reconectar()
            except Exception as e:
                logger.exception("Error inesperado en loop backend: %s", e)
                await asyncio.sleep(1)

            if time.time() - ultima_metrica >= 120:
                self._log_metricas_periodicas()
                self.pipeline.log_metricas()
                ultima_metrica = time.time()

    # ----------------------------------------------------------
    # Procesamiento de mensajes
    # ----------------------------------------------------------

    async def _procesar_mensaje(self, msg_id: bytes, campos: dict) -> None:
        """
        Pipeline completo para un mensaje de texto:
        Deserializar → PipelineService → Publicar RespuestaTTS → ACK
        """
        t0 = time.perf_counter()
        logger.debug("BackendConsumer: procesando mensaje %s…", msg_id)

        try:
            # 1. Deserializar TranscripcionTexto
            transcripcion = self._deserializar_transcripcion(campos)
            if transcripcion is None:
                self._ack(msg_id)
                self._errores += 1
                return

            # 2. Verificar delivery count (poison pill detection)
            delivery_count = self._obtener_delivery_count(msg_id)
            if delivery_count > MAX_DELIVERY_COUNT:
                self._enviar_dlq(msg_id, campos, f"delivery_count={delivery_count}")
                self._ack(msg_id)
                self._al_dlq += 1
                return

            # 3. Ejecutar pipeline (async: Gemini + RAG + SQL)
            respuesta_tts = await self.pipeline.procesar(transcripcion)

            if respuesta_tts is None:
                logger.error("Pipeline retornó None para msg_id=%s", msg_id)
                self._errores += 1
                # No ACK: se reintentará
                return

            # 4. Publicar RespuestaTTS al stream de salida
            exito = self._publicar_respuesta_tts(respuesta_tts)

            if exito:
                self._ack(msg_id)
                self._procesados += 1
                latencia = (time.perf_counter() - t0) * 1000
                logger.info(
                    "✅ [%d] Backend completado en %.0fms | prioridad=%s | session=%s",
                    self._procesados,
                    latencia,
                    respuesta_tts.prioridad_tts,
                    respuesta_tts.session_id,
                )
            else:
                self._errores += 1
                logger.warning("No se pudo publicar RespuestaTTS para msg_id=%s", msg_id)

        except Exception as e:
            self._errores += 1
            logger.exception("Error procesando msg_id=%s: %s", msg_id, e)

    def _deserializar_transcripcion(self, campos: dict) -> Optional[TranscripcionTexto]:
        try:
            payload_json = campos.get(FIELD_TEXT_PAYLOAD)
            if not payload_json:
                logger.error("Mensaje sin campo '%s'.", FIELD_TEXT_PAYLOAD)
                return None
            return TranscripcionTexto.model_validate_json(payload_json)
        except Exception as e:
            logger.error("Error deserializando TranscripcionTexto: %s", e)
            return None

    def _publicar_respuesta_tts(self, respuesta: RespuestaTTS) -> bool:
        try:
            self._client.xadd(
                name=self.stream_respuesta,
                fields={
                    FIELD_RESP_PAYLOAD:    respuesta.model_dump_json(),
                    FIELD_RESP_SESSION_ID: respuesta.session_id,
                },
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            return True
        except Exception as e:
            logger.error("Error publicando RespuestaTTS: %s", e)
            return False

    def _ack(self, msg_id: bytes) -> None:
        try:
            self._client.xack(self.stream_texto, self.group_name, msg_id)
        except Exception as e:
            logger.warning("Error en XACK: %s", e)

    def _enviar_dlq(self, msg_id: bytes, campos: dict, razon: str) -> None:
        try:
            self._client.xadd(
                name=STREAM_DLQ,
                fields={
                    "msg_id_original": str(msg_id),
                    "stream_origen":   self.stream_texto,
                    "razon":           razon,
                    "consumer":        self.consumer_name,
                    "payload":         json.dumps(campos),
                },
                maxlen=1000,
            )
            logger.warning("Mensaje %s enviado al DLQ. Razón: %s", msg_id, razon)
        except Exception as e:
            logger.error("Error enviando al DLQ: %s", e)

    def _obtener_delivery_count(self, msg_id: bytes) -> int:
        try:
            pel = self._client.xpending_range(
                self.stream_texto, self.group_name,
                min=msg_id, max=msg_id, count=1,
            )
            if pel:
                return pel[0].get("times_delivered", 1)
        except Exception:
            pass
        return 1

    # ----------------------------------------------------------
    # Consumer Group
    # ----------------------------------------------------------

    def _asegurar_grupo_consumer(self) -> None:
        try:
            self._client.xgroup_create(
                name=self.stream_texto,
                groupname=self.group_name,
                id="0",
                mkstream=True,
            )
            logger.info(
                "Consumer group '%s' creado en stream '%s'.",
                self.group_name, self.stream_texto,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("Consumer group '%s' ya existe.", self.group_name)
            else:
                raise

    async def _reclamar_mensajes_pendientes(self) -> None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            pendientes = await loop.run_in_executor(
                None,
                lambda: self._client.xpending(self.stream_texto, self.group_name),
            )
            total = pendientes.get("pending", 0)
            if total == 0:
                return

            logger.warning(
                "Encontrados %d mensaje(s) pendiente(s) en PEL. Reclamando…", total
            )
            reclamados = await loop.run_in_executor(
                None,
                lambda: self._client.xautoclaim(
                    name=self.stream_texto,
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    min_idle_time=CLAIM_IDLE_MS,
                    start_id="0-0",
                    count=100,
                ),
            )
            mensajes = reclamados[1] if reclamados else []
            for msg_id, campos in mensajes:
                await self._procesar_mensaje(msg_id, campos)
        except Exception as e:
            logger.warning("Error reclamando mensajes pendientes: %s", e)

    # ----------------------------------------------------------
    # Conexión Redis
    # ----------------------------------------------------------

    def _conectar(self) -> None:
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
                logger.info(
                    "✅ BackendConsumer conectado a Redis (intento %d).", intento
                )
                return
            except Exception as e:
                logger.warning(
                    "Intento %d conexión Redis fallido: %s. Reintentando en %.0fs…",
                    intento, e, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

    def _reconectar(self) -> None:
        self._client = None
        self._conectar()
        self._asegurar_grupo_consumer()

    # ----------------------------------------------------------
    # Métricas
    # ----------------------------------------------------------

    def _log_metricas_periodicas(self) -> None:
        logger.info(
            "📊 BackendConsumer '%s' — procesados: %d | errores: %d | al_dlq: %d",
            self.consumer_name, self._procesados, self._errores, self._al_dlq,
        )

    def _log_metricas_finales(self) -> None:
        logger.info(
            "🛑 BackendConsumer detenido. Procesados: %d | Errores: %d | DLQ: %d",
            self._procesados, self._errores, self._al_dlq,
        )
