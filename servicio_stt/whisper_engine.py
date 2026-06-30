"""
servicio_stt/whisper_engine.py
================================
Motor de transcripción de voz usando faster-whisper (CTranslate2).

Responsabilidades:
  - Cargar el modelo Whisper UNA SOLA VEZ al inicio del servicio.
  - Transcribir audio PCM 16kHz (numpy float32) a texto en español.
  - Forzar el idioma a 'es' para evitar detección incorrecta en entornos ruidosos.
  - Medir y exponer latencia de inferencia para monitoreo.
  - Manejar errores de transcripción sin crashear el servicio.

Modelo recomendado: 'small' — equilibrio óptimo para agricultura:
  - Vocabulario técnico agronómico (plagas, cultivos, insumos)
  - Acento mexicano rural (fonética diferente al español estándar)
  - Latencia < 2s para audio de 3-5s en CPU modesta

Referencia: https://github.com/SYSTRAN/faster-whisper
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig:
    """Configuración del motor Whisper. Cargada desde variables de entorno."""
    model_name:    str   = "small"       # tiny | base | small | medium | large-v3
    device:        str   = "cpu"         # cpu | cuda
    compute_type:  str   = "int8"        # int8 | float16 | float32
    # Directorio de caché del modelo (descargado de HuggingFace en primer uso)
    download_root: Optional[str] = "/models/whisper"
    # Parámetros de transcripción
    language:      str   = "es"          # Forzar español — crítico para ruido rural
    beam_size:     int   = 5             # Mayor = más preciso, más lento
    best_of:       int   = 5
    temperature:   float = 0.0           # 0.0 = greedy (más rápido y determinista)
    # VAD interno de faster-whisper (doble capa de filtrado)
    vad_filter:    bool  = True
    vad_min_silence_duration_ms: int = 500
    # Tokens iniciales que guían al modelo para vocabulario agronómico
    initial_prompt: str = (
        "Transcripción de un agricultor mexicano hablando sobre cultivos, plagas, "
        "cosechas, fertilizantes y trámites de cooperativa agrícola. "
        "Nombres de cultivos: maíz, frijol, café, trigo, sorgo. "
        "Insumos: urea, fertilizante, herbicida, fungicida, insecticida."
    )


@dataclass
class TranscripcionResultado:
    """Resultado de la transcripción de un utterance."""
    texto:            str
    idioma_detectado: str
    confianza:        float          # Probabilidad promedio de los segmentos
    latencia_ms:      float          # Tiempo de inferencia
    segmentos:        list[dict] = field(default_factory=list)
    es_vacio:         bool = False   # True si no se detectó habla real


class WhisperEngine:
    """
    Motor de transcripción Whisper con faster-whisper (CTranslate2).

    Singleton de facto: se instancia una vez y se reutiliza para todos
    los utterances del pipeline, amortizando el costo de carga del modelo.

    Ejemplo de uso:
        engine = WhisperEngine(WhisperConfig(model_name='small'))
        engine.cargar_modelo()
        resultado = engine.transcribir(audio_float32_array)
        print(resultado.texto)
    """

    def __init__(self, config: WhisperConfig):
        self.config = config
        self._model: Optional[WhisperModel] = None
        self._modelo_cargado = False
        self._total_transcripciones = 0
        self._latencia_acumulada_ms = 0.0
        self._errores = 0

    # ----------------------------------------------------------
    # Ciclo de vida
    # ----------------------------------------------------------

    def cargar_modelo(self) -> None:
        """
        Descarga (si es necesario) y carga el modelo Whisper en memoria.
        Primera ejecución: descarga desde HuggingFace (~460MB para 'small').
        Ejecuciones posteriores: carga desde caché local (2-5s).
        """
        if self._modelo_cargado:
            return

        logger.info(
            "Cargando modelo Whisper '%s' (device=%s, compute_type=%s)…",
            self.config.model_name,
            self.config.device,
            self.config.compute_type,
        )
        t0 = time.perf_counter()

        try:
            self._model = WhisperModel(
                model_size_or_path=self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=self.config.download_root,
                # Número de workers para CTranslate2 (1 es suficiente para CPU embebida)
                cpu_threads=4,
                num_workers=1,
            )
            self._modelo_cargado = True
            latencia = (time.perf_counter() - t0) * 1000
            logger.info(
                "✅ Modelo Whisper '%s' cargado en %.0fms.",
                self.config.model_name,
                latencia,
            )
        except Exception as e:
            raise RuntimeError(
                f"No se pudo cargar el modelo Whisper '{self.config.model_name}': {e}\n"
                "Verificar:\n"
                "  1. Conexión a internet (primer arranque descarga el modelo)\n"
                f"  2. Espacio en disco en {self.config.download_root}\n"
                "  3. Que compute_type='int8' sea soportado en este hardware"
            ) from e

    # ----------------------------------------------------------
    # Transcripción
    # ----------------------------------------------------------

    def transcribir(self, audio: np.ndarray) -> TranscripcionResultado:
        """
        Transcribe un utterance de audio a texto en español.

        Args:
            audio: Array numpy float32, rango [-1, 1], 16kHz, mono.
                   Mismo formato que produce VADProcessor.

        Returns:
            TranscripcionResultado con texto, confianza y métricas.

        Raises:
            RuntimeError: si el modelo no fue cargado con cargar_modelo().
        """
        if not self._modelo_cargado or self._model is None:
            raise RuntimeError("Llamar a cargar_modelo() antes de transcribir().")

        if len(audio) == 0:
            return TranscripcionResultado(
                texto="", idioma_detectado="es", confianza=0.0,
                latencia_ms=0.0, es_vacio=True,
            )

        t0 = time.perf_counter()

        try:
            segments_iter, info = self._model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                initial_prompt=self.config.initial_prompt,
                vad_filter=self.config.vad_filter,
                vad_parameters={
                    "min_silence_duration_ms": self.config.vad_min_silence_duration_ms
                },
                word_timestamps=False,  # No necesarios para nuestro caso de uso
                condition_on_previous_text=False,  # Cada utterance es independiente
            )

            # Materializar el iterador (faster-whisper es lazy)
            segmentos = []
            textos = []
            confianzas = []

            for seg in segments_iter:
                segmentos.append({
                    "inicio": seg.start,
                    "fin": seg.end,
                    "texto": seg.text.strip(),
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                })
                # Filtrar segmentos con alta probabilidad de no-voz
                if seg.no_speech_prob < 0.6:
                    textos.append(seg.text.strip())
                    # Convertir log-probabilidad a probabilidad lineal aproximada
                    prob = min(1.0, max(0.0, 1.0 + seg.avg_logprob / 5.0))
                    confianzas.append(prob)

            latencia_ms = (time.perf_counter() - t0) * 1000
            self._latencia_acumulada_ms += latencia_ms
            self._total_transcripciones += 1

            texto_final = " ".join(textos).strip()
            confianza_prom = float(np.mean(confianzas)) if confianzas else 0.0
            es_vacio = len(texto_final) == 0

            if es_vacio:
                logger.debug(
                    "Utterance transcrito como vacío (solo ruido/silencio). "
                    "Latencia: %.0fms", latencia_ms
                )
            else:
                logger.info(
                    "STT: '%s…' (confianza=%.2f, latencia=%.0fms, idioma=%s)",
                    texto_final[:60],
                    confianza_prom,
                    latencia_ms,
                    info.language,
                )

            return TranscripcionResultado(
                texto=texto_final,
                idioma_detectado=info.language,
                confianza=confianza_prom,
                latencia_ms=latencia_ms,
                segmentos=segmentos,
                es_vacio=es_vacio,
            )

        except Exception as e:
            self._errores += 1
            latencia_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "Error de transcripción (intento %d, %.0fms): %s",
                self._errores, latencia_ms, e,
            )
            return TranscripcionResultado(
                texto="",
                idioma_detectado="es",
                confianza=0.0,
                latencia_ms=latencia_ms,
                es_vacio=True,
            )

    # ----------------------------------------------------------
    # Métricas
    # ----------------------------------------------------------

    @property
    def latencia_promedio_ms(self) -> float:
        if self._total_transcripciones == 0:
            return 0.0
        return self._latencia_acumulada_ms / self._total_transcripciones

    @property
    def total_transcripciones(self) -> int:
        return self._total_transcripciones

    @property
    def total_errores(self) -> int:
        return self._errores

    def log_metricas(self) -> None:
        logger.info(
            "📊 Whisper STT — transcripciones: %d | errores: %d | latencia_prom: %.0fms",
            self._total_transcripciones,
            self._errores,
            self.latencia_promedio_ms,
        )
