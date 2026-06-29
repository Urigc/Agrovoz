"""
shared/schemas.py
=================
Fuente única de verdad para todos los esquemas Pydantic del pipeline AgroVoz.

Importado por:
  - cliente_ingesta    → AudioChunk (payload publicado a Redis)
  - servicio_stt       → TranscripcionTexto (payload producido tras STT)
  - backend_api        → DiagnosticoAgronomico, RegistroEntrega, SolicitudInsumo
  - servicio_tts       → RespuestaTTS (payload consumido para síntesis)

NOTA: Este archivo vive en shared/ y se monta como volumen read-only
en todos los contenedores bajo /app/shared/schemas.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ===========================================================
# PIPELINE: Capa 1 — Audio a Redis Stream "agrovoz:audio"
# Producido por: cliente_ingesta
# Consumido por: servicio_stt
# ===========================================================

class AudioChunk(BaseModel):
    """
    Payload que el cliente_ingesta publica en el Stream de Redis
    cada vez que Silero VAD detecta un utterance completo de voz.

    El audio viaja como bytes codificados en base64 para ser
    serializable en los campos de texto de Redis Streams.
    """
    chunk_id:        str       = Field(default_factory=lambda: str(uuid.uuid4()))
    audio_b64:       str       = Field(..., description="Audio PCM 16kHz, mono, 16-bit, codificado en base64.")
    sample_rate:     int       = Field(default=16000)
    duracion_ms:     float     = Field(..., description="Duración del utterance en milisegundos.")
    timestamp_utc:   str       = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 UTC cuando se capturó el audio."
    )
    # Trazabilidad entre etapas del pipeline
    session_id:      str       = Field(..., description="ID de sesión del agricultor en este dispositivo.")
    device_id:       str       = Field(default="cliente_default", description="Identificador del dispositivo de campo.")

    @field_validator("duracion_ms")
    @classmethod
    def validar_duracion(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("duracion_ms debe ser positivo.")
        if v > 60_000:
            raise ValueError("Utterance demasiado largo (> 60s). Verificar VAD.")
        return v


# ===========================================================
# PIPELINE: Capa 2 — Texto a Redis Stream "agrovoz:texto"
# Producido por: servicio_stt
# Consumido por: backend_api
# ===========================================================

class TranscripcionTexto(BaseModel):
    """
    Resultado del STT (Whisper). Encadena el chunk de audio
    con su transcripción en texto para que el backend lo procese.
    """
    transcripcion_id:  str    = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id:          str    = Field(..., description="Referencia al AudioChunk original.")
    session_id:        str
    texto:             str    = Field(..., description="Texto transcrito por Whisper en español.")
    idioma_detectado:  str    = Field(default="es")
    confianza_stt:     float  = Field(default=1.0, ge=0.0, le=1.0)
    timestamp_utc:     str    = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @field_validator("texto")
    @classmethod
    def validar_texto(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("La transcripción no puede estar vacía.")
        return v


# ===========================================================
# PIPELINE: Capa 3 — Respuesta al Stream "agrovoz:respuesta"
# Producido por: backend_api
# Consumido por: servicio_tts
# ===========================================================

class RespuestaTTS(BaseModel):
    """
    Payload que el backend envía al servicio TTS para síntesis de voz.
    El campo 'prioridad_tts' decide qué motor usar (Piper vs ElevenLabs).
    """
    respuesta_id:     str    = Field(default_factory=lambda: str(uuid.uuid4()))
    transcripcion_id: str
    session_id:       str
    texto_para_leer:  str    = Field(..., description="Texto que el TTS debe sintetizar.")
    # 'rapida' → Piper local (< 200ms), 'premium' → ElevenLabs (más natural)
    prioridad_tts:    Literal["rapida", "premium"] = "rapida"
    timestamp_utc:    str   = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ===========================================================
# ESQUEMAS DE SALIDA ESTRUCTURADA DE GEMINI
# El backend_api obliga a Gemini a responder en estos formatos.
# ===========================================================

class DiagnosticoAgronomico(BaseModel):
    """
    Respuesta de Gemini para consultas de diagnóstico de plagas/enfermedades.
    Se construye con contexto RAG de pgvector.
    """
    intencion:                   Literal["consulta_agronomica"]
    plaga_o_enfermedad_probable: str     = Field(..., description="Nombre científico y común.")
    nivel_confianza:             float   = Field(..., ge=0.0, le=1.0)
    tratamiento_recomendado:     str
    advertencias:                list[str] = Field(default_factory=list)
    # Texto en español natural, empático, que el TTS leerá al agricultor
    respuesta_para_tts:          str     = Field(
        ...,
        description="Texto de voz: claro, sin tecnicismos excesivos, en segunda persona."
    )
    # Fragmentos RAG que fundamentaron la respuesta (para auditoría)
    fuentes_rag:                 list[str] = Field(default_factory=list)


class RegistroEntrega(BaseModel):
    """
    Respuesta de Gemini para intenciones de entrega de cosecha.
    Los datos son extraídos del dictado del agricultor.
    """
    intencion:            Literal["registrar_entrega"]
    id_membresco_socio:   str
    tipo_cultivo:         str
    cantidad_kg:          float   = Field(..., gt=0)
    calidad_estimada:     Optional[str] = None         # Si el agricultor lo menciona
    confirmacion_para_tts: str


class SolicitudInsumo(BaseModel):
    """
    Respuesta de Gemini para solicitudes de insumos al almacén.
    """
    intencion:             Literal["solicitar_insumo"]
    id_membresco_socio:    str
    nombre_insumo:         str     = Field(..., description="Nombre como fue dictado, puede ser informal.")
    cantidad:              float   = Field(..., gt=0)
    unidad_medida:         str
    confirmacion_para_tts: str


class IntentionMixta(BaseModel):
    """
    Cuando el agricultor dicta múltiples acciones en un solo utterance.
    Ej: 'entrego 800 kilos de frijol y necesito 5 kilos de urea'.
    El backend procesa cada intención en orden secuencial.
    """
    intencion:    Literal["mixta"]
    sub_intentos: list[RegistroEntrega | SolicitudInsumo] = Field(
        ..., description="Lista de intenciones individuales en orden de mención."
    )
    confirmacion_para_tts: str


# ===========================================================
# ENUM HELPERS (usados para validación y routing)
# ===========================================================

class TipoIntencion:
    AGRONOMICA = "consulta_agronomica"
    ENTREGA    = "registrar_entrega"
    INSUMO     = "solicitar_insumo"
    MIXTA      = "mixta"
