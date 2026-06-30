"""
backend_api/routers/agronomico.py
====================================
Router FastAPI para consultas agronómicas directas vía HTTP.

Complementa el pipeline de voz: los operadores o técnicos de la cooperativa
pueden hacer consultas directamente por HTTP sin pasar por el audio.

Endpoints:
  POST /agronomico/diagnostico   → Diagnóstico de plaga/enfermedad con RAG
  GET  /agronomico/conocimiento  → Lista fragmentos de conocimiento disponibles

Útil también para:
  - Debug del sistema RAG durante desarrollo
  - Interfaz web para técnicos agrónomos
  - Integración con sistemas externos de la cooperativa
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import RAGRepository, get_session
from services.embedding_service import get_embedding_service
from services.gemini_service import GeminiService
from shared.schemas import DiagnosticoAgronomico

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agronomico", tags=["Agronómico"])


# ===========================================================
# Modelos de request/response HTTP
# ===========================================================

class ConsultaAgronomicaRequest(BaseModel):
    texto:          str   = Field(..., min_length=10, max_length=1000,
                                   description="Descripción del problema del agricultor.")
    tipo_cultivo:   Optional[str] = Field(None, description="Filtrar RAG por cultivo. Ej: 'maiz'")
    top_k:          int   = Field(default=3, ge=1, le=10,
                                   description="Número de fragmentos RAG a recuperar.")

    model_config = {"json_schema_extra": {
        "example": {
            "texto": "Las hojas de mi maíz tienen manchas amarillas con polvo negro en el envés",
            "tipo_cultivo": "maiz",
            "top_k": 3,
        }
    }}


class FragmentoRAGResponse(BaseModel):
    id:                  int
    titulo_documento:    str
    tipo_cultivo_aplica: Optional[str]
    fragmento_texto:     str
    similitud:           float
    metadata:            dict


class DiagnosticoHTTPResponse(BaseModel):
    diagnostico:    DiagnosticoAgronomico
    fragmentos_rag: list[FragmentoRAGResponse]
    latencia_rag_ms: Optional[float] = None


# ===========================================================
# Dependency injection
# ===========================================================

def get_gemini_service_dep() -> GeminiService:
    """Obtiene la instancia global de GeminiService (inyectada en app startup)."""
    from main import gemini_service_instance
    if gemini_service_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio Gemini no disponible.",
        )
    return gemini_service_instance


# ===========================================================
# Endpoints
# ===========================================================

@router.post(
    "/diagnostico",
    response_model=DiagnosticoHTTPResponse,
    summary="Diagnóstico agronómico con RAG",
    description=(
        "Recibe la descripción de un problema del agricultor, busca fragmentos "
        "relevantes en la base de conocimiento agronómico (pgvector) y genera "
        "un diagnóstico usando Gemini con el contexto recuperado."
    ),
)
async def diagnostico_agronomico(
    request: ConsultaAgronomicaRequest,
    gemini: GeminiService = Depends(get_gemini_service_dep),
) -> DiagnosticoHTTPResponse:
    """
    Endpoint principal de diagnóstico agronómico.

    1. Genera embedding del texto de la consulta
    2. Busca fragmentos similares en pgvector (RAG)
    3. Llama a Gemini con el contexto RAG
    4. Retorna el diagnóstico estructurado
    """
    import time
    t0 = time.perf_counter()

    # 1. Generar embedding de la consulta
    try:
        embedding_svc = get_embedding_service()
        embedding = embedding_svc.generar(request.texto)
    except Exception as e:
        logger.error("Error generando embedding: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en servicio de embeddings: {str(e)}",
        )

    # 2. Búsqueda RAG en pgvector
    fragmentos = []
    latencia_rag = None
    try:
        t_rag = time.perf_counter()
        async with get_session() as session:
            fragmentos = await RAGRepository.buscar_fragmentos(
                session=session,
                embedding=embedding,
                top_k=request.top_k,
                cultivo_filtro=request.tipo_cultivo,
                min_similarity=0.25,   # Umbral más bajo en API directa (más contexto)
            )
        latencia_rag = (time.perf_counter() - t_rag) * 1000
        logger.info(
            "RAG: %d fragmento(s) en %.0fms para consulta '%s…'",
            len(fragmentos), latencia_rag, request.texto[:50],
        )
    except Exception as e:
        logger.error("Error en búsqueda RAG: %s", e)
        # Continuar sin RAG (Gemini responderá con conocimiento propio)

    # 3. Diagnóstico con Gemini
    try:
        resultado = await gemini.procesar_transcripcion(
            texto=request.texto,
            contexto_rag=fragmentos,
        )
    except Exception as e:
        logger.error("Error en Gemini: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error en servicio de IA: {str(e)}",
        )

    if not isinstance(resultado, DiagnosticoAgronomico):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Gemini no clasificó la consulta como agronómica. "
                   "Asegúrate de describir un problema de cultivo o plaga.",
        )

    latencia_total = (time.perf_counter() - t0) * 1000
    logger.info(
        "✅ Diagnóstico completado en %.0fms: %s (confianza=%.2f)",
        latencia_total,
        resultado.plaga_o_enfermedad_probable,
        resultado.nivel_confianza,
    )

    return DiagnosticoHTTPResponse(
        diagnostico=resultado,
        fragmentos_rag=[
            FragmentoRAGResponse(
                id=int(f.get("id", 0)),
                titulo_documento=str(f.get("titulo_documento", "")),
                tipo_cultivo_aplica=f.get("tipo_cultivo_aplica"),
                fragmento_texto=str(f.get("fragmento_texto", "")),
                similitud=float(f.get("similitud", 0.0)),
                metadata=dict(f.get("metadata") or {}),
            )
            for f in fragmentos
        ],
        latencia_rag_ms=latencia_rag,
    )


@router.get(
    "/conocimiento",
    summary="Lista fragmentos de conocimiento agronómico",
    description="Retorna los fragmentos de conocimiento disponibles en la base RAG, con filtros opcionales.",
)
async def listar_conocimiento(
    cultivo:  Optional[str] = None,
    limite:   int = 20,
) -> dict:
    """Lista fragmentos de la base de conocimiento RAG para auditoría y debug."""
    try:
        async with get_session() as session:
            from sqlalchemy import text
            query = "SELECT id, titulo_documento, tipo_cultivo_aplica, seccion, metadata FROM conocimiento_agronomico"
            params: dict = {}
            if cultivo:
                query += " WHERE tipo_cultivo_aplica = :cultivo"
                params["cultivo"] = cultivo
            query += f" ORDER BY id LIMIT {min(limite, 100)}"

            result = await session.execute(text(query), params)
            fragmentos = [dict(row._mapping) for row in result]

        return {
            "total": len(fragmentos),
            "filtro_cultivo": cultivo,
            "fragmentos": fragmentos,
        }
    except Exception as e:
        logger.error("Error listando conocimiento: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
