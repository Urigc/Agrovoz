"""
backend_api/services/embedding_service.py
==========================================
Servicio de generación de embeddings para el sistema RAG.

Motor: sentence-transformers con modelo multilingüe que soporta español.
Modelo por defecto: paraphrase-multilingual-mpnet-base-v2
  - 768 dims nativos, proyectados a 1536 para compatibilidad con el schema
  - Soporta español, inglés y 49 idiomas más
  - ~420MB en disco, ~1.5GB en RAM
  - Latencia: ~50ms por query en CPU modesta

Ventaja sobre usar la API de Gemini para embeddings:
  - Cero costo (local, sin límites de cuota)
  - Sin dependencia de internet para inferencia
  - Latencia determinista (no depende de red)

NOTA: Los embeddings de los documentos de la base de conocimiento se generan
UNA SOLA VEZ al cargar los documentos (script de seed/ingesta).
En producción, solo se genera el embedding de la CONSULTA del agricultor en
tiempo real. Esto hace el pipeline muy eficiente.
"""

import logging
import time
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Servicio de embeddings usando sentence-transformers.

    Singleton: se instancia una vez al arrancar el backend y se reutiliza.
    El modelo se carga en memoria en el primer uso (lazy loading).
    """

    def __init__(
        self,
        model_name:    str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        target_dim:    int = 1536,
        device:        str = "cpu",
        cache_dir:     Optional[str] = "/models/embeddings",
    ):
        self.model_name = model_name
        self.target_dim = target_dim
        self.device     = device
        self.cache_dir  = cache_dir

        self._model = None
        self._model_dim: Optional[int] = None
        self._total_embeddings = 0
        self._latencia_acumulada_ms = 0.0

        logger.info(
            "EmbeddingService creado: modelo='%s', target_dim=%d",
            model_name, target_dim,
        )

    def cargar_modelo(self) -> None:
        """Carga el modelo de sentence-transformers en memoria."""
        if self._model is not None:
            return

        logger.info("Cargando modelo de embeddings '%s'…", self.model_name)
        t0 = time.perf_counter()

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            # Determinar dimensión nativa del modelo
            test_vec = self._model.encode(["test"], show_progress_bar=False)
            self._model_dim = test_vec.shape[1]

            latencia = (time.perf_counter() - t0) * 1000
            logger.info(
                "✅ Modelo de embeddings cargado: dim=%d, latencia=%.0fms",
                self._model_dim, latencia,
            )
        except Exception as e:
            raise RuntimeError(
                f"No se pudo cargar el modelo de embeddings '{self.model_name}': {e}"
            ) from e

    def generar(self, texto: str) -> list[float]:
        """
        Genera el embedding de un texto en español.

        Args:
            texto: Consulta del agricultor o fragmento de documento.

        Returns:
            Vector de 'target_dim' dimensiones como lista de floats.
        """
        if self._model is None:
            self.cargar_modelo()

        t0 = time.perf_counter()

        # Preprocesamiento mínimo: limpiar y truncar
        texto_limpio = texto.strip()[:512]   # sentence-transformers recomienda max 512 tokens

        vector = self._model.encode(
            [texto_limpio],
            normalize_embeddings=True,   # Normalización L2 para similitud coseno correcta
            show_progress_bar=False,
            batch_size=1,
        )[0]  # shape: (model_dim,)

        # Proyectar al target_dim del schema SQL (padding o truncado)
        vector_proyectado = self._proyectar(vector)

        latencia = (time.perf_counter() - t0) * 1000
        self._total_embeddings += 1
        self._latencia_acumulada_ms += latencia

        logger.debug(
            "Embedding generado: texto='%s…', latencia=%.0fms",
            texto_limpio[:40], latencia,
        )

        return vector_proyectado.tolist()

    def generar_batch(self, textos: list[str]) -> list[list[float]]:
        """
        Genera embeddings para múltiples textos en un solo forward pass.
        Más eficiente que llamadas individuales para carga masiva de documentos.
        """
        if self._model is None:
            self.cargar_modelo()

        if not textos:
            return []

        textos_limpios = [t.strip()[:512] for t in textos]

        vectores = self._model.encode(
            textos_limpios,
            normalize_embeddings=True,
            show_progress_bar=len(textos) > 10,
            batch_size=32,
        )  # shape: (n, model_dim)

        self._total_embeddings += len(textos)
        return [self._proyectar(v).tolist() for v in vectores]

    def _proyectar(self, vector: np.ndarray) -> np.ndarray:
        """
        Proyecta el vector nativo del modelo a la dimensión target del schema SQL.

        Estrategia:
          - Si model_dim < target_dim: padding con ceros (al final del vector)
          - Si model_dim > target_dim: truncado (primeras target_dim dimensiones)
          - Si model_dim == target_dim: sin cambios

        El padding mantiene la integridad de la información del vector original.
        Los embeddings de documentos y queries deben proyectarse de la MISMA forma.
        """
        model_dim = len(vector)
        if model_dim == self.target_dim:
            return vector
        elif model_dim < self.target_dim:
            # Padding con ceros
            proyectado = np.zeros(self.target_dim, dtype=np.float32)
            proyectado[:model_dim] = vector
            return proyectado
        else:
            # Truncar
            return vector[:self.target_dim].astype(np.float32)

    @property
    def latencia_promedio_ms(self) -> float:
        if self._total_embeddings == 0:
            return 0.0
        return self._latencia_acumulada_ms / self._total_embeddings

    @property
    def modelo_cargado(self) -> bool:
        return self._model is not None


# ===========================================================
# Instancia global (se inicializa en el lifespan de FastAPI)
# ===========================================================
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Retorna la instancia global del servicio de embeddings."""
    global _embedding_service
    if _embedding_service is None:
        raise RuntimeError(
            "EmbeddingService no inicializado. "
            "Verificar que init_embedding_service() fue llamado en el lifespan."
        )
    return _embedding_service


def init_embedding_service(
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    target_dim: int = 1536,
    cache_dir:  Optional[str] = "/models/embeddings",
) -> EmbeddingService:
    """Inicializa el servicio global. Llamar en el lifespan de FastAPI."""
    global _embedding_service
    _embedding_service = EmbeddingService(
        model_name=model_name,
        target_dim=target_dim,
        cache_dir=cache_dir,
    )
    _embedding_service.cargar_modelo()
    return _embedding_service
