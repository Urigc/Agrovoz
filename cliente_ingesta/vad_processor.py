"""
cliente_ingesta/vad_processor.py
==================================
Procesador de detección de actividad de voz (Voice Activity Detection) usando
el modelo Silero VAD ejecutado localmente con ONNX Runtime.

Responsabilidades:
  - Cargar el modelo Silero VAD (ONNX) UNA SOLA VEZ al inicio.
  - Consumir chunks de PCM crudo de AudioCapture.
  - Detectar segmentos de voz vs silencio en tiempo real.
  - Emitir utterances completos (arrays numpy de audio) cuando detecta
    un final de discurso (silencio sostenido post-voz).

El VAD actúa como "filtro inteligente": solo propaga hacia Redis el audio
que realmente contiene voz, eliminando silencios y ruido de fondo.
Esto es crítico para minimizar el ancho de banda en zonas rurales.

Basado en: https://github.com/snakers4/silero-vad
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Parámetros de configuración del VAD. Cargados desde .env."""
    sample_rate:       int   = 16_000
    threshold:         float = 0.5     # Probabilidad mínima para considerar voz
    min_silence_ms:    int   = 700     # Silencio mínimo para cerrar utterance
    min_speech_ms:     int   = 250     # Duración mínima de voz para no descartar
    pre_speech_pad_ms: int   = 200     # Audio pre-voz a incluir (contexto)

    @property
    def chunk_size(self) -> int:
        """Silero VAD funciona con ventanas de 512 samples a 16kHz."""
        return 512

    @property
    def min_silence_samples(self) -> int:
        return int(self.min_silence_ms * self.sample_rate / 1000)

    @property
    def min_speech_samples(self) -> int:
        return int(self.min_speech_ms * self.sample_rate / 1000)

    @property
    def pre_speech_pad_samples(self) -> int:
        return int(self.pre_speech_pad_ms * self.sample_rate / 1000)


@dataclass
class Utterance:
    """Un segmento de audio que contiene voz detectada."""
    audio_pcm:      np.ndarray   # PCM 16kHz, float32 normalizado [-1, 1]
    duracion_ms:    float
    timestamp_ini:  float = field(default_factory=time.time)
    confianza_prom: float = 0.0  # Probabilidad promedio de voz del VAD


class VADProcessor:
    """
    Procesador de Silero VAD que transforma un stream de chunks PCM crudo
    en utterances discretos y completos de voz del agricultor.

    Máquina de estados internos:
        SILENCIO → (detecta voz) → EN_VOZ → (silencio prolongado) → EMITE_UTTERANCE
                                                                           ↓
                                                                      SILENCIO

    El modelo Silero VAD (ONNX) se descarga automáticamente desde torch.hub
    en el primer uso y se cachea localmente en /root/.cache/torch/.
    """

    def __init__(self, config: VADConfig):
        self.config = config
        self._model: Optional[object] = None
        self._model_loaded = False

        # Estado interno de la máquina de estados
        self._estado: str = "SILENCIO"    # 'SILENCIO' | 'EN_VOZ'
        self._buffer_voz: list[np.ndarray] = []
        self._samples_silencio: int = 0
        self._confianzas_voz: list[float] = []

        # Buffer pre-voz circular: guarda los últimos N samples antes de la voz
        # para incluir como contexto (evita cortar el inicio de palabras)
        self._pre_speech_buffer: list[np.ndarray] = []
        self._pre_speech_max_chunks = max(
            1, config.pre_speech_pad_samples // config.chunk_size
        )

        logger.info(
            "VADProcessor inicializado: threshold=%.2f, min_silence=%dms, "
            "min_speech=%dms, pre_pad=%dms",
            config.threshold,
            config.min_silence_ms,
            config.min_speech_ms,
            config.pre_speech_pad_ms,
        )

    # ----------------------------------------------------------
    # Ciclo de vida
    # ----------------------------------------------------------

    def cargar_modelo(self) -> None:
        """
        Descarga y carga el modelo Silero VAD desde torch.hub.
        Se debe llamar una vez antes de procesar audio.
        Puede tardar 10-30s en la primera ejecución (descarga del modelo).
        """
        if self._model_loaded:
            return

        logger.info("Cargando modelo Silero VAD…")
        inicio = time.time()
        try:
            # Silero VAD está en torch.hub como un modelo oficial
            # Se descarga a ~/.cache/torch/hub/ en el primer uso
            self._model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,   # Usar ONNX para mayor velocidad en CPU
            )
            # Desempaquetar utilidades (no usamos get_speech_ts directamente)
            self._get_speech_timestamps = utils[0]

            self._model_loaded = True
            logger.info(
                "✅ Modelo Silero VAD cargado en %.2fs.", time.time() - inicio
            )
        except Exception as e:
            raise RuntimeError(
                f"No se pudo cargar Silero VAD: {e}. "
                "Verificar conexión a internet en el primer arranque o "
                "que el modelo esté cacheado en /root/.cache/torch/hub/"
            ) from e

    def reset_estado(self) -> None:
        """Reinicia la máquina de estados (útil entre sesiones de un agricultor)."""
        self._estado = "SILENCIO"
        self._buffer_voz.clear()
        self._samples_silencio = 0
        self._confianzas_voz.clear()
        self._pre_speech_buffer.clear()
        logger.debug("Estado del VAD reiniciado.")

    # ----------------------------------------------------------
    # Procesamiento principal
    # ----------------------------------------------------------

    def procesar_chunk(self, raw_audio: bytes) -> Optional[Utterance]:
        """
        Procesa un chunk de PCM crudo (bytes) y aplica el VAD.

        Args:
            raw_audio: Bytes de PCM 16-bit, 16kHz, mono.

        Returns:
            Utterance completo si se detectó fin de discurso, None si no.
        """
        if not self._model_loaded:
            raise RuntimeError("Llamar a cargar_modelo() antes de procesar_chunk().")

        # Convertir PCM int16 bytes → float32 numpy [-1, 1]
        audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Procesar en ventanas del tamaño que espera Silero VAD (512 samples)
        utterance = None
        for i in range(0, len(audio_float), self.config.chunk_size):
            window = audio_float[i: i + self.config.chunk_size]
            if len(window) < self.config.chunk_size:
                # Rellenar con ceros si el último chunk es incompleto
                window = np.pad(window, (0, self.config.chunk_size - len(window)))

            resultado = self._procesar_ventana(window)
            if resultado is not None:
                utterance = resultado  # Solo puede haber uno por chunk grande

        return utterance

    def procesar_stream(
        self, chunks_iter: Generator[bytes, None, None]
    ) -> Generator[Utterance, None, None]:
        """
        Procesa un iterador de chunks y genera utterances conforme se completan.
        Útil para integración con AudioCapture.

        Ejemplo:
            for utterance in vad.procesar_stream(capture.iter_chunks()):
                publicar_a_redis(utterance)
        """
        for raw_audio in chunks_iter:
            resultado = self.procesar_chunk(raw_audio)
            if resultado is not None:
                yield resultado

    # ----------------------------------------------------------
    # Máquina de estados interna
    # ----------------------------------------------------------

    def _procesar_ventana(self, audio_window: np.ndarray) -> Optional[Utterance]:
        """
        Aplica Silero VAD a una ventana de 512 samples y actualiza estado.
        Retorna un Utterance cuando se completa un segmento de voz.
        """
        # Inferencia VAD: probabilidad de que la ventana contenga voz
        audio_tensor = torch.from_numpy(audio_window).unsqueeze(0)  # [1, 512]
        with torch.no_grad():
            prob_voz: float = self._model(audio_tensor, self.config.sample_rate).item()

        es_voz = prob_voz >= self.config.threshold

        if self._estado == "SILENCIO":
            return self._manejar_silencio(audio_window, es_voz, prob_voz)
        else:  # EN_VOZ
            return self._manejar_en_voz(audio_window, es_voz, prob_voz)

    def _manejar_silencio(
        self, window: np.ndarray, es_voz: bool, prob: float
    ) -> Optional[Utterance]:
        """Estado SILENCIO: esperando inicio de voz."""
        # Mantener buffer circular pre-voz
        self._pre_speech_buffer.append(window)
        if len(self._pre_speech_buffer) > self._pre_speech_max_chunks:
            self._pre_speech_buffer.pop(0)

        if es_voz:
            # TRANSICIÓN: SILENCIO → EN_VOZ
            logger.debug("VAD: inicio de voz detectado (p=%.3f)", prob)
            self._estado = "EN_VOZ"
            self._samples_silencio = 0
            self._confianzas_voz = [prob]

            # Incluir buffer pre-voz para no cortar el inicio de palabras
            self._buffer_voz = list(self._pre_speech_buffer)
            self._pre_speech_buffer.clear()

        return None  # En silencio no emitimos nada

    def _manejar_en_voz(
        self, window: np.ndarray, es_voz: bool, prob: float
    ) -> Optional[Utterance]:
        """Estado EN_VOZ: acumulando audio de voz."""
        self._buffer_voz.append(window)
        self._confianzas_voz.append(prob)

        if es_voz:
            # Continúa la voz — resetear contador de silencio
            self._samples_silencio = 0
        else:
            # Silencio dentro de posible utterance
            self._samples_silencio += self.config.chunk_size

            if self._samples_silencio >= self.config.min_silence_samples:
                # TRANSICIÓN: EN_VOZ → SILENCIO + emitir utterance
                return self._finalizar_utterance()

        return None

    def _finalizar_utterance(self) -> Optional[Utterance]:
        """
        Concatena el buffer de voz, valida duración mínima y
        retorna el Utterance listo para publicar a Redis.
        """
        self._estado = "SILENCIO"

        # Concatenar todos los chunks del utterance
        audio_completo = np.concatenate(self._buffer_voz)

        # Limpiar estado
        self._buffer_voz.clear()
        confianzas = list(self._confianzas_voz)
        self._confianzas_voz.clear()
        self._samples_silencio = 0

        # Verificar duración mínima de voz
        duracion_ms = len(audio_completo) / self.config.sample_rate * 1000

        if len(audio_completo) < self.config.min_speech_samples:
            logger.debug(
                "VAD: utterance descartado por ser muy corto (%.0fms < %dms mínimo).",
                duracion_ms, self.config.min_speech_ms,
            )
            return None

        confianza_prom = float(np.mean(confianzas)) if confianzas else 0.0
        logger.info(
            "VAD: utterance detectado — %.0fms, confianza promedio=%.3f",
            duracion_ms, confianza_prom,
        )

        return Utterance(
            audio_pcm=audio_completo,
            duracion_ms=duracion_ms,
            confianza_prom=confianza_prom,
        )
