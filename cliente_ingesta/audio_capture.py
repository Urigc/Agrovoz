"""
cliente_ingesta/audio_capture.py
=================================
Captura de audio crudo del micrófono en formato PCM 16kHz, mono, 16-bit.

Responsabilidades:
  - Abrir el dispositivo de audio del sistema operativo.
  - Emitir frames de audio de tamaño fijo a través de una cola thread-safe.
  - Gestionar el ciclo de vida del stream de audio (abrir/cerrar/reiniciar).
  - NO hace procesamiento de voz: eso es tarea del VAD processor.

Formato de salida: bytes de PCM crudo, 16kHz, 16-bit, mono.
Este es el formato nativo que requiere Whisper para la transcripción.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Parámetros de captura de audio. Leer desde .env vía AudioCapture."""
    sample_rate:    int = 16_000    # Hz — requerido por Whisper y Silero VAD
    chunk_frames:   int = 8_000     # Frames por chunk (0.5 s a 16kHz)
    channels:       int = 1         # Mono
    format:         int = pyaudio.paInt16  # 16-bit PCM
    device_index:   Optional[int] = None   # None = dispositivo por defecto del sistema

    @property
    def chunk_duration_ms(self) -> float:
        return (self.chunk_frames / self.sample_rate) * 1000

    @property
    def bytes_per_frame(self) -> int:
        # paInt16 = 2 bytes por frame
        return 2


class AudioCapture:
    """
    Captura audio del micrófono y deposita chunks de PCM en una cola.

    Thread-safe: el audio se captura en un hilo daemon separado.
    El hilo principal (VAD) consume de la cola.

    Ejemplo de uso:
        config = AudioConfig()
        capture = AudioCapture(config)
        capture.start()
        try:
            while True:
                chunk = capture.get_chunk(timeout=1.0)
                if chunk:
                    process(chunk)
        finally:
            capture.stop()
    """

    def __init__(
        self,
        config: AudioConfig,
        queue_maxsize: int = 50,  # Máximo de chunks en buffer (50 × 0.5s = 25s de buffer)
    ):
        self.config = config
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_maxsize)
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._frames_captured: int = 0
        self._frames_dropped: int = 0   # Cuando la cola está llena

    # ----------------------------------------------------------
    # Ciclo de vida público
    # ----------------------------------------------------------

    def start(self) -> None:
        """Abre el micrófono y comienza a capturar audio en un hilo daemon."""
        if self._is_running:
            logger.warning("AudioCapture ya está ejecutándose.")
            return

        logger.info(
            "Iniciando captura de audio: %d Hz, %d frames/chunk (%.0f ms), canal mono.",
            self.config.sample_rate,
            self.config.chunk_frames,
            self.config.chunk_duration_ms,
        )

        self._stop_event.clear()
        self._pyaudio = pyaudio.PyAudio()

        # Listar y loggear dispositivos disponibles (útil para debug en campo)
        self._log_audio_devices()

        self._stream = self._open_stream()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="agrovoz-audio-capture",
            daemon=True,  # El hilo muere cuando el proceso principal termina
        )
        self._capture_thread.start()
        self._is_running = True
        logger.info("✅ Captura de audio iniciada correctamente.")

    def stop(self) -> None:
        """Detiene la captura de audio y libera recursos de hardware."""
        logger.info("Deteniendo captura de audio…")
        self._stop_event.set()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)

        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.warning("Error al cerrar stream de audio: %s", e)
            self._stream = None

        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception as e:
                logger.warning("Error al terminar PyAudio: %s", e)
            self._pyaudio = None

        self._is_running = False
        logger.info(
            "Captura detenida. Frames capturados: %d, descartados: %d.",
            self._frames_captured,
            self._frames_dropped,
        )

    def get_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Obtiene el siguiente chunk de audio de la cola.
        Retorna None si el timeout expira sin datos (permite chequear _stop_event).
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def queue_size(self) -> int:
        return self._audio_queue.qsize()

    # ----------------------------------------------------------
    # Implementación interna
    # ----------------------------------------------------------

    def _open_stream(self) -> pyaudio.Stream:
        """Abre el stream de PyAudio con reintento ante error de dispositivo."""
        attempts = 0
        max_attempts = 5
        retry_delay = 2.0

        while attempts < max_attempts:
            try:
                stream = self._pyaudio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.config.chunk_frames,
                    input_device_index=self.config.device_index,
                    # Callback = None: modo bloqueante (más simple y estable en ARM)
                )
                logger.info(
                    "Stream de audio abierto en dispositivo índice=%s.",
                    self.config.device_index or "default",
                )
                return stream
            except OSError as e:
                attempts += 1
                logger.error(
                    "Error al abrir dispositivo de audio (intento %d/%d): %s",
                    attempts, max_attempts, e,
                )
                if attempts < max_attempts:
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"No se pudo abrir el dispositivo de audio después de {max_attempts} intentos. "
            "Verificar que el micrófono esté conectado y accesible (grupo 'audio')."
        )

    def _capture_loop(self) -> None:
        """
        Bucle principal de captura. Corre en hilo daemon.
        Lee chunks del stream de PyAudio y los deposita en la cola.
        """
        logger.debug("Hilo de captura de audio iniciado.")

        while not self._stop_event.is_set():
            try:
                # Lectura bloqueante: espera hasta tener chunk_frames disponibles
                raw_audio = self._stream.read(
                    self.config.chunk_frames,
                    exception_on_overflow=False,  # Evita crash por buffer overflow
                )

                self._frames_captured += 1

                # Depositar en cola; si está llena, descartar chunk más antiguo
                if self._audio_queue.full():
                    try:
                        self._audio_queue.get_nowait()  # Descartar el más viejo
                        self._frames_dropped += 1
                        logger.warning(
                            "Cola de audio llena (%d slots). Chunk descartado. "
                            "El sistema de procesamiento puede estar saturado.",
                            self._audio_queue.maxsize,
                        )
                    except queue.Empty:
                        pass

                self._audio_queue.put_nowait(raw_audio)

            except OSError as e:
                if not self._stop_event.is_set():
                    logger.error("Error de lectura de audio: %s. Reintentando en 1s…", e)
                    time.sleep(1.0)
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.exception("Error inesperado en captura de audio: %s", e)
                    time.sleep(0.5)

        logger.debug("Hilo de captura de audio finalizado.")

    def _log_audio_devices(self) -> None:
        """Lista dispositivos de audio disponibles para facilitar configuración en campo."""
        if not self._pyaudio:
            return
        try:
            count = self._pyaudio.get_device_count()
            logger.info("Dispositivos de audio disponibles (%d):", count)
            for i in range(count):
                info = self._pyaudio.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    logger.info(
                        "  [%d] %s — %d canales entrada, %.0f Hz",
                        i,
                        info.get("name", "Desconocido"),
                        info.get("maxInputChannels", 0),
                        info.get("defaultSampleRate", 0),
                    )
        except Exception as e:
            logger.warning("No se pudo listar dispositivos de audio: %s", e)
