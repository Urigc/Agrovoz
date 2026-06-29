"""
cliente_ingesta/redis_publisher.py
=====================================
Publicador resiliente de utterances de audio al Redis Stream "agrovoz:audio".

Responsabilidades:
  - Conectar a Redis con reintentos exponenciales (crítico en zonas rurales
    con conectividad intermitente).
  - Serializar Utterances en JSON y publicarlos al stream.
  - Mantener el stream con MAXLEN para evitar crecimiento ilimitado en memoria.
  - Exponer métricas simples de publicación (mensajes enviados/fallidos).
  - Manejar desconexiones silenciosamente: encolar en buffer local y reintentar.

Resiliencia: si Redis no está disponible, los mensajes se guardan en un
deque en memoria (buffer local) y se reenvían cuando la conexión se restaura.
Esto permite que el agricultor siga hablando aunque el backend caiga momentáneamente.
"""

import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from shared.redis_config import (
    FIELD_AUDIO_PAYLOAD,
    FIELD_AUDIO_SESSION_ID,
    STREAM_AUDIO,
    STREAM_MAXLEN,
)
from shared.schemas import AudioChunk
from vad_processor import Utterance

logger = logging.getLogger(__name__)


@dataclass
class PublisherConfig:
    """Parámetros del publicador. Cargados desde variables de entorno."""
    redis_url:          str   = "redis://localhost:6379/0"
    stream_name:        str   = STREAM_AUDIO
    stream_maxlen:      int   = STREAM_MAXLEN

    # Resiliencia — reintentos de conexión
    max_reconnect_attempts: int   = 0      # 0 = reintentar indefinidamente
    reconnect_delay_base:   float = 1.0    # Segundos de espera inicial
    reconnect_delay_max:    float = 60.0   # Tope de espera (backoff exponencial)
    reconnect_delay_factor: float = 2.0    # Factor de crecimiento exponencial

    # Buffer local en memoria cuando Redis no está disponible
    # Si el agricultor habla 10 utterances mientras Redis cae (1-2 min),
    # se almacenan aquí y se reenvían al reconectar.
    local_buffer_maxlen: int = 100


@dataclass
class PublisherMetrics:
    """Métricas de operación del publicador."""
    mensajes_enviados:  int = 0
    mensajes_fallidos:  int = 0
    reconexiones:       int = 0
    mensajes_buffered:  int = 0    # Acumulados en buffer local durante desconexión
    mensajes_flushed:   int = 0    # Buffer local vaciado exitosamente


class RedisPublisher:
    """
    Publicador resiliente de AudioChunks al Redis Stream.

    Gestiona la conexión, serialización, reintentos y buffer local
    de forma transparente para el llamador.

    Uso:
        pub = RedisPublisher(config, session_id="sesion_001", device_id="campo_A")
        pub.conectar()
        pub.publicar(utterance)
        pub.desconectar()
    """

    def __init__(
        self,
        config: PublisherConfig,
        session_id: str,
        device_id: str = "cliente_default",
    ):
        self.config = config
        self.session_id = session_id
        self.device_id = device_id

        self._redis_client: Optional[redis.Redis] = None
        self._conectado: bool = False
        self._metrics = PublisherMetrics()

        # Buffer local: deque con maxlen para evitar consumo excesivo de RAM
        # Si se llena, descarta los más antiguos (mensajes de menor prioridad)
        self._local_buffer: deque[dict] = deque(maxlen=config.local_buffer_maxlen)

        logger.info(
            "RedisPublisher creado → stream='%s', session='%s', device='%s'",
            config.stream_name, session_id, device_id,
        )

    # ----------------------------------------------------------
    # Ciclo de vida
    # ----------------------------------------------------------

    def conectar(self) -> None:
        """
        Establece la conexión a Redis con reintentos exponenciales.
        Bloquea hasta conectar o hasta agotar max_reconnect_attempts.
        """
        logger.info("Conectando a Redis: %s", self._redis_url_seguro())
        self._conectar_con_reintentos()

    def desconectar(self) -> None:
        """Cierra la conexión a Redis y loggea métricas finales."""
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                pass
            self._redis_client = None
        self._conectado = False

        logger.info(
            "RedisPublisher desconectado. Métricas finales: "
            "enviados=%d, fallidos=%d, reconexiones=%d, buffered=%d, flushed=%d",
            self._metrics.mensajes_enviados,
            self._metrics.mensajes_fallidos,
            self._metrics.reconexiones,
            self._metrics.mensajes_buffered,
            self._metrics.mensajes_flushed,
        )

    # ----------------------------------------------------------
    # Publicación principal
    # ----------------------------------------------------------

    def publicar(self, utterance: Utterance) -> bool:
        """
        Serializa un Utterance y lo publica al Redis Stream.

        Si Redis no está disponible, guarda en buffer local y retorna False.
        El buffer local se vacía automáticamente en la próxima publicación exitosa.

        Returns:
            True si se publicó exitosamente, False si fue al buffer local.
        """
        # Construir el payload AudioChunk
        payload = self._construir_payload(utterance)

        # Si hay buffer local pendiente, intentar vaciarlo primero
        if self._local_buffer:
            self._flush_buffer_local()

        # Intentar publicar
        exito = self._publicar_payload(payload)
        if not exito:
            # Guardar en buffer local para reintento posterior
            self._local_buffer.append(payload)
            self._metrics.mensajes_buffered += 1
            logger.warning(
                "Utterance buffered localmente (buffer_size=%d). "
                "Redis no disponible.",
                len(self._local_buffer),
            )

        return exito

    # ----------------------------------------------------------
    # Implementación interna
    # ----------------------------------------------------------

    def _construir_payload(self, utterance: Utterance) -> dict:
        """
        Convierte un Utterance en el diccionario de campos para Redis XADD.

        El audio se serializa como base64 para ser almacenable en Redis Streams
        (que maneja valores como strings/bytes, no arrays numpy).
        """
        # Convertir float32 → int16 → bytes → base64
        audio_int16 = (utterance.audio_pcm * 32768.0).clip(-32768, 32767).astype("int16")
        audio_bytes = audio_int16.tobytes()
        audio_b64   = base64.b64encode(audio_bytes).decode("ascii")

        chunk = AudioChunk(
            audio_b64=audio_b64,
            duracion_ms=utterance.duracion_ms,
            session_id=self.session_id,
            device_id=self.device_id,
        )

        return {
            FIELD_AUDIO_PAYLOAD:    chunk.model_dump_json(),
            FIELD_AUDIO_SESSION_ID: self.session_id,
        }

    def _publicar_payload(self, campos: dict) -> bool:
        """Publica un diccionario de campos al stream. Retorna True si éxito."""
        if not self._conectado or not self._redis_client:
            self._intentar_reconexion()
            if not self._conectado:
                return False

        try:
            msg_id = self._redis_client.xadd(
                name=self.config.stream_name,
                fields=campos,
                maxlen=self.config.stream_maxlen,
                approximate=True,   # MAXLEN ~ N: más eficiente, tolerado en nuestro caso
            )
            self._metrics.mensajes_enviados += 1
            logger.debug(
                "Utterance publicado → stream='%s', msg_id='%s', duracion=%.0fms",
                self.config.stream_name,
                msg_id,
                # Extraer duracion_ms del JSON para el log
                json.loads(campos[FIELD_AUDIO_PAYLOAD]).get("duracion_ms", 0),
            )
            return True

        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.error("Conexión Redis perdida al publicar: %s", e)
            self._conectado = False
            self._metrics.mensajes_fallidos += 1
            return False

        except Exception as e:
            logger.exception("Error inesperado al publicar en Redis: %s", e)
            self._metrics.mensajes_fallidos += 1
            return False

    def _flush_buffer_local(self) -> None:
        """
        Intenta publicar todos los mensajes acumulados en el buffer local.
        Se invoca automáticamente al detectar que Redis volvió a estar disponible.
        """
        if not self._conectado:
            return

        flushed = 0
        while self._local_buffer:
            payload = self._local_buffer[0]   # Peekar sin quitar
            exito = self._publicar_payload(payload)
            if exito:
                self._local_buffer.popleft()
                flushed += 1
            else:
                # Redis volvió a caer, dejar el resto para después
                break

        if flushed:
            self._metrics.mensajes_flushed += flushed
            logger.info(
                "Buffer local vaciado: %d mensaje(s) reenviados a Redis. "
                "Pendientes: %d.",
                flushed, len(self._local_buffer),
            )

    def _conectar_con_reintentos(self) -> None:
        """Intenta conectar a Redis con backoff exponencial."""
        delay = self.config.reconnect_delay_base
        intento = 0

        while True:
            intento += 1
            try:
                client = redis.from_url(
                    self.config.redis_url,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                    decode_responses=True,   # Trabajar con strings, no bytes
                )
                # Verificar conexión real
                client.ping()

                self._redis_client = client
                self._conectado = True

                logger.info(
                    "✅ Conectado a Redis exitosamente (intento %d).", intento
                )

                # Asegurar que el stream existe (XADD lo crea implícitamente,
                # pero verificar el grupo de consumidores downstream no es responsabilidad aquí)
                return

            except (RedisConnectionError, RedisTimeoutError, Exception) as e:
                self._conectado = False
                logger.warning(
                    "Intento %d de conexión a Redis fallido: %s. "
                    "Reintentando en %.0fs…",
                    intento, e, delay,
                )

                # Verificar límite de reintentos
                if (
                    self.config.max_reconnect_attempts > 0
                    and intento >= self.config.max_reconnect_attempts
                ):
                    raise RuntimeError(
                        f"No se pudo conectar a Redis después de {intento} intentos."
                    )

                time.sleep(delay)
                # Backoff exponencial con tope máximo
                delay = min(delay * self.config.reconnect_delay_factor, self.config.reconnect_delay_max)

    def _intentar_reconexion(self) -> None:
        """Intento rápido (no bloqueante) de reconexión a Redis."""
        logger.info("Intentando reconexión a Redis…")
        try:
            self._conectar_con_reintentos()
            if self._conectado:
                self._metrics.reconexiones += 1
        except RuntimeError as e:
            logger.error("Reconexión fallida: %s", e)

    def _redis_url_seguro(self) -> str:
        """Retorna la URL de Redis con la contraseña enmascarada para logs."""
        url = self.config.redis_url
        if "@" in url:
            # redis://:password@host:port/db → redis://:***@host:port/db
            prefix = url[:url.index("//") + 2]
            rest   = url[url.index("@"):]
            return f"{prefix}:***{rest}"
        return url

    @property
    def metricas(self) -> PublisherMetrics:
        return self._metrics

    @property
    def conectado(self) -> bool:
        return self._conectado

    @property
    def buffer_local_size(self) -> int:
        return len(self._local_buffer)
