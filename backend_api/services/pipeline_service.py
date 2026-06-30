"""
backend_api/services/pipeline_service.py
==========================================
Orquestador principal del pipeline de procesamiento de AgroVoz.

Este servicio es el cerebro del backend. Para cada transcripción recibida:

  Flujo A — Consulta Agronómica (RAG):
    1. Generar embedding de la consulta
    2. Buscar fragmentos relevantes en pgvector (RAG)
    3. Llamar a Gemini con el contexto recuperado
    4. Publicar diagnóstico al stream TTS

  Flujo B — Trámite de Cooperativa (Transaccional):
    1. Llamar a Gemini para extraer entidades (socio, cultivo, cantidades)
    2. Validar el socio en PostgreSQL
    3. Ejecutar transacciones SQL (INSERT entregas / solicitudes)
    4. Publicar confirmación al stream TTS

  Flujo C — Intención Mixta:
    Combina A y B en secuencia, consolidando las confirmaciones.

Garantías de atomicidad:
  - Las transacciones SQL usan la session de SQLAlchemy con rollback automático.
  - El ACK al stream Redis solo ocurre tras publicar exitosamente al stream TTS.
"""

import logging
import time
from typing import Optional

from db.database import (
    AsyncSession,
    EntregaCosechaRepository,
    InsumoRepository,
    RAGRepository,
    SolicitudInsumoRepository,
    SocioRepository,
    get_session,
)
from services.embedding_service import EmbeddingService
from services.gemini_service import GeminiService, RespuestaGemini
from shared.schemas import (
    DiagnosticoAgronomico,
    IntentionMixta,
    RegistroEntrega,
    RespuestaTTS,
    SolicitudInsumo,
    TipoIntencion,
    TranscripcionTexto,
)

logger = logging.getLogger(__name__)


class PipelineService:
    """
    Orquestador del pipeline backend de AgroVoz.

    Instanciado una vez al arrancar y reutilizado para todas las transcripciones.
    Gestiona el flujo completo desde texto hasta respuesta TTS.
    """

    def __init__(
        self,
        gemini_service:    GeminiService,
        embedding_service: EmbeddingService,
        rag_top_k:         int   = 3,
        rag_min_similarity: float = 0.3,
    ):
        self.gemini     = gemini_service
        self.embeddings = embedding_service
        self.rag_top_k  = rag_top_k
        self.rag_min_similarity = rag_min_similarity

        # Métricas
        self._procesados       = 0
        self._errores          = 0
        self._flujo_agronomico = 0
        self._flujo_entrega    = 0
        self._flujo_insumo     = 0
        self._flujo_mixto      = 0

    async def procesar(self, transcripcion: TranscripcionTexto) -> Optional[RespuestaTTS]:
        """
        Procesa una transcripción de texto y retorna la respuesta para TTS.

        Args:
            transcripcion: Modelo Pydantic con el texto transcrito y metadata.

        Returns:
            RespuestaTTS con el texto a sintetizar, o None si hubo error irrecuperable.
        """
        t0 = time.perf_counter()
        logger.info(
            "Pipeline: procesando '%s…' (session=%s)",
            transcripcion.texto[:60],
            transcripcion.session_id,
        )

        try:
            # Obtener contexto RAG si la consulta parece agronómica
            # (optimización: evitar embedding para consultas de trámites)
            contexto_rag = await self._obtener_contexto_rag_si_aplica(
                transcripcion.texto
            )

            # Llamar a Gemini para clasificar y extraer entidades
            respuesta_gemini = await self.gemini.procesar_transcripcion(
                texto=transcripcion.texto,
                contexto_rag=contexto_rag,
            )

            if respuesta_gemini is None:
                logger.error("Gemini no pudo procesar: '%s'", transcripcion.texto[:80])
                return self._respuesta_error_generica(transcripcion)

            # Ejecutar la lógica de negocio según la intención
            texto_tts = await self._ejecutar_intencion(respuesta_gemini)

            latencia = (time.perf_counter() - t0) * 1000
            self._procesados += 1
            logger.info(
                "✅ Pipeline completado en %.0fms. TTS: '%s…'",
                latencia, texto_tts[:60],
            )

            # Decidir si usar Piper (rápido) o ElevenLabs (premium)
            prioridad = self._decidir_prioridad_tts(respuesta_gemini, texto_tts)

            return RespuestaTTS(
                transcripcion_id=transcripcion.transcripcion_id,
                session_id=transcripcion.session_id,
                texto_para_leer=texto_tts,
                prioridad_tts=prioridad,
            )

        except Exception as e:
            self._errores += 1
            logger.exception(
                "Error en pipeline para '%s': %s",
                transcripcion.texto[:60], e,
            )
            return self._respuesta_error_generica(transcripcion)

    # ----------------------------------------------------------
    # RAG: Recuperación de contexto agronómico
    # ----------------------------------------------------------

    async def _obtener_contexto_rag_si_aplica(
        self, texto: str
    ) -> list[dict]:
        """
        Genera el embedding de la consulta y busca fragmentos relevantes en pgvector.
        Retorna lista vacía si no hay contexto relevante (consultas de trámite).
        """
        # Heurística rápida: si el texto menciona plantas/plagas, buscar RAG
        palabras_agronomicas = {
            "plaga", "hongo", "insecto", "hoja", "mancha", "amarill", "negro",
            "podr", "enfermedad", "roya", "hongo", "cult", "maiz", "frijol",
            "cafe", "planta", "trat", "fumigar", "fertiliz", "semilla", "cosech",
            "florac", "raiz", "stem", "tallo", "fruto", "grano", "maleza",
            "hierba", "yerbas", "gusano", "chapul", "mosca", "pulgon",
        }

        texto_lower = texto.lower()
        parece_agronomico = any(p in texto_lower for p in palabras_agronomicas)

        if not parece_agronomico:
            logger.debug("Texto sin palabras agronómicas, omitiendo búsqueda RAG.")
            return []

        try:
            # Generar embedding de la consulta
            embedding = self.embeddings.generar(texto)

            # Buscar en pgvector
            async with get_session() as session:
                fragmentos = await RAGRepository.buscar_fragmentos(
                    session=session,
                    embedding=embedding,
                    top_k=self.rag_top_k,
                    min_similarity=self.rag_min_similarity,
                )

            logger.info(
                "RAG: %d fragmento(s) recuperado(s) con similitud >= %.2f",
                len(fragmentos), self.rag_min_similarity,
            )
            return fragmentos

        except Exception as e:
            logger.error("Error en búsqueda RAG: %s. Continuando sin contexto.", e)
            return []

    # ----------------------------------------------------------
    # Ejecución de lógica de negocio por intención
    # ----------------------------------------------------------

    async def _ejecutar_intencion(self, respuesta: RespuestaGemini) -> str:
        """Despacha la lógica de negocio según el tipo de intención."""
        if isinstance(respuesta, DiagnosticoAgronomico):
            self._flujo_agronomico += 1
            return await self._ejecutar_diagnostico(respuesta)

        elif isinstance(respuesta, RegistroEntrega):
            self._flujo_entrega += 1
            return await self._ejecutar_entrega(respuesta)

        elif isinstance(respuesta, SolicitudInsumo):
            self._flujo_insumo += 1
            return await self._ejecutar_solicitud_insumo(respuesta)

        elif isinstance(respuesta, IntentionMixta):
            self._flujo_mixto += 1
            return await self._ejecutar_mixta(respuesta)

        return "Lo siento, no entendí tu solicitud. ¿Puedes repetirlo?"

    async def _ejecutar_diagnostico(self, diag: DiagnosticoAgronomico) -> str:
        """Para consultas agronómicas, la respuesta ya viene de Gemini con RAG."""
        # La respuesta TTS ya fue generada por Gemini con el contexto RAG
        return diag.respuesta_para_tts

    async def _ejecutar_entrega(self, entrega: RegistroEntrega) -> str:
        """Registra una entrega de cosecha en PostgreSQL."""
        async with get_session() as session:
            # 1. Validar que el socio existe
            socio = await SocioRepository.buscar_por_membresco(
                session, entrega.id_membresco_socio
            )
            if not socio:
                return (
                    f"No encontré al socio con número {entrega.id_membresco_socio}. "
                    "Por favor verifica tu número de membresía."
                )

            # 2. Registrar entrega
            entrega_id = await EntregaCosechaRepository.registrar(
                session=session,
                socio_id=socio.id,
                tipo_cultivo=entrega.tipo_cultivo,
                cantidad_kg=entrega.cantidad_kg,
                calidad=entrega.calidad_estimada,
            )

            if entrega_id < 0:
                return (
                    "Hubo un error al registrar tu entrega. "
                    "Por favor habla con el operador del almacén."
                )

            logger.info(
                "Entrega registrada: socio=%s, cultivo=%s, kg=%.1f, id=%d",
                entrega.id_membresco_socio,
                entrega.tipo_cultivo,
                entrega.cantidad_kg,
                entrega_id,
            )

        # Usar la confirmación generada por Gemini (más natural que un template)
        return entrega.confirmacion_para_tts

    async def _ejecutar_solicitud_insumo(self, solicitud: SolicitudInsumo) -> str:
        """Crea una solicitud de insumo en PostgreSQL con verificación de stock."""
        async with get_session() as session:
            # 1. Validar socio
            socio = await SocioRepository.buscar_por_membresco(
                session, solicitud.id_membresco_socio
            )
            if not socio:
                return (
                    f"No encontré al socio con número {solicitud.id_membresco_socio}. "
                    "Por favor verifica tu número de membresía."
                )

            # 2. Buscar insumo por nombre fuzzy
            insumos_candidatos = await InsumoRepository.buscar_por_nombre_fuzzy(
                session, solicitud.nombre_insumo
            )
            if not insumos_candidatos:
                return (
                    f"No encontré el insumo '{solicitud.nombre_insumo}' en el inventario. "
                    "Pregunta al almacenista qué insumos hay disponibles."
                )

            insumo = insumos_candidatos[0]   # El más similar por trigrama

            # 3. Verificar stock suficiente
            tiene_stock, stock_actual = await InsumoRepository.verificar_stock(
                session, insumo["id"], solicitud.cantidad
            )
            if not tiene_stock:
                return (
                    f"Lo siento, no hay suficiente {insumo['nombre_insumo']} en el almacén. "
                    f"Solo quedan {stock_actual:.1f} {insumo['unidad_medida']}. "
                    "Tu solicitud quedó pendiente para cuando llegue más."
                )

            # 4. Crear solicitud en estado 'Pendiente'
            solicitud_id = await SolicitudInsumoRepository.crear(
                session=session,
                socio_id=socio.id,
                insumo_id=insumo["id"],
                cantidad_solicitada=solicitud.cantidad,
            )

            logger.info(
                "Solicitud creada: socio=%s, insumo='%s', cantidad=%.1f %s, id=%d",
                solicitud.id_membresco_socio,
                insumo["nombre_insumo"],
                solicitud.cantidad,
                solicitud.unidad_medida,
                solicitud_id,
            )

        return solicitud.confirmacion_para_tts

    async def _ejecutar_mixta(self, mixta: IntentionMixta) -> str:
        """Procesa múltiples intenciones secuencialmente."""
        confirmaciones = []

        for sub in mixta.sub_intentos:
            if isinstance(sub, RegistroEntrega):
                self._flujo_entrega += 1
                texto = await self._ejecutar_entrega(sub)
            elif isinstance(sub, SolicitudInsumo):
                self._flujo_insumo += 1
                texto = await self._ejecutar_solicitud_insumo(sub)
            else:
                texto = "No pude procesar una de tus solicitudes."

            confirmaciones.append(texto)

        # Consolidar en una respuesta fluida
        if len(confirmaciones) == 1:
            return confirmaciones[0]
        return " Además, ".join(confirmaciones) + "."

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _decidir_prioridad_tts(
        self, respuesta: RespuestaGemini, texto_tts: str
    ) -> str:
        """
        Decide si usar Piper (rápido, local) o ElevenLabs (premium, API).

        Lógica:
          - Diagnósticos agronómicos complejos → premium (más natural para explicaciones)
          - Confirmaciones de trámites → rapida (simple, < 30 palabras)
          - Respuestas largas (> 50 palabras) → premium
        """
        palabras = len(texto_tts.split())
        if isinstance(respuesta, DiagnosticoAgronomico) and palabras > 30:
            return "premium"
        if palabras > 50:
            return "premium"
        return "rapida"

    def _respuesta_error_generica(self, transcripcion: TranscripcionTexto) -> RespuestaTTS:
        """Respuesta de fallback cuando ocurre un error irrecuperable."""
        return RespuestaTTS(
            transcripcion_id=transcripcion.transcripcion_id,
            session_id=transcripcion.session_id,
            texto_para_leer=(
                "Lo siento, tuve un problema al procesar tu consulta. "
                "Por favor intenta de nuevo o habla con el operador."
            ),
            prioridad_tts="rapida",
        )

    def log_metricas(self) -> None:
        logger.info(
            "📊 Pipeline — procesados: %d | errores: %d | "
            "agronómica: %d | entrega: %d | insumo: %d | mixta: %d",
            self._procesados, self._errores,
            self._flujo_agronomico, self._flujo_entrega,
            self._flujo_insumo, self._flujo_mixto,
        )
