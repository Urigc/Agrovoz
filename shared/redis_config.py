"""
shared/redis_config.py
======================
Configuración centralizada del bus de eventos Redis Streams.

Define los nombres de streams, grupos de consumidores y constantes
de forma que sean consistentes entre todos los servicios del pipeline.

Todos los servicios importan de aquí: NUNCA hardcodear nombres de streams.

Relación de streams:
    cliente_ingesta → [agrovoz:audio] → servicio_stt
    servicio_stt    → [agrovoz:texto] → backend_api
    backend_api     → [agrovoz:respuesta] → servicio_tts
"""

import os


# ===========================================================
# NOMBRES DE STREAMS (configurables vía .env)
# ===========================================================

STREAM_AUDIO     = os.getenv("REDIS_STREAM_AUDIO",    "agrovoz:audio")
STREAM_TEXTO     = os.getenv("REDIS_STREAM_TEXTO",    "agrovoz:texto")
STREAM_RESPUESTA = os.getenv("REDIS_STREAM_RESPUESTA","agrovoz:respuesta")

# Stream auxiliar para métricas de latencia del pipeline
STREAM_METRICAS  = "agrovoz:metricas"

# Stream para errores y mensajes muertos (Dead Letter Queue)
STREAM_DLQ       = "agrovoz:dlq"

# ===========================================================
# GRUPOS DE CONSUMIDORES
# ===========================================================

GROUP_STT     = os.getenv("REDIS_GROUP_STT",     "grupo_stt")
GROUP_BACKEND = os.getenv("REDIS_GROUP_BACKEND", "grupo_backend")
GROUP_TTS     = os.getenv("REDIS_GROUP_TTS",     "grupo_tts")

# ===========================================================
# CONSTANTES DE COMPORTAMIENTO
# ===========================================================

# Tiempo máximo que un mensaje puede estar "en vuelo" antes de ser
# reclamado por otro consumidor del grupo (en milisegundos)
CLAIM_IDLE_MS = 30_000   # 30 segundos

# Número máximo de mensajes a leer por iteración de consumer
MAX_MESSAGES_PER_POLL = 1

# Tiempo de bloqueo del XREADGROUP cuando no hay mensajes (ms)
# 0 = bloqueo indefinido, None = no bloquear
BLOCK_MS = 5_000   # 5 segundos

# Máxima longitud del stream (MAXLEN) — evita crecimiento ilimitado
# Los mensajes más viejos son descartados automáticamente
STREAM_MAXLEN = 10_000

# Tiempo de retención máximo de un mensaje en ms (para auditoría)
# Pasado este tiempo, se puede eliminar
MESSAGE_TTL_MS = 3_600_000   # 1 hora

# ===========================================================
# CAMPOS DE PAYLOAD EN REDIS STREAMS
# Redis Streams almacena campos clave-valor como strings.
# Estos son los nombres canónicos de los campos en cada stream.
# ===========================================================

# Campos del stream agrovoz:audio
FIELD_AUDIO_PAYLOAD    = "payload_json"   # JSON serializado de AudioChunk
FIELD_AUDIO_SESSION_ID = "session_id"

# Campos del stream agrovoz:texto
FIELD_TEXT_PAYLOAD     = "payload_json"   # JSON serializado de TranscripcionTexto
FIELD_TEXT_SESSION_ID  = "session_id"

# Campos del stream agrovoz:respuesta
FIELD_RESP_PAYLOAD     = "payload_json"   # JSON serializado de RespuestaTTS
FIELD_RESP_SESSION_ID  = "session_id"
