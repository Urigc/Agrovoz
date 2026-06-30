"""
backend_api/main.py
=====================
Aplicación FastAPI principal del backend AgroVoz.

Responsabilidades:
  - Gestionar el lifespan: arrancar/apagar todos los servicios ordenadamente
  - Registrar los routers HTTP (agronomico, cooperativa, health)
  - Arrancar el BackendConsumer como tarea de fondo (asyncio.Task)
  - Exponer instancias de servicios para inyección de dependencias

El backend opera en DOS modos simultáneos:
  1. Modo HTTP (FastAPI): recibe peticiones de operadores/técnicos
  2. Modo Worker (BackendConsumer): procesa transcripciones del pipeline de voz

Ambos modos comparten el mismo pool de conexiones PostgreSQL y los
mismos servicios de Gemini y embeddings.
"""

import asyncio
import logging
import os
import signal
import socket
import sys
from contextlib import asynccontextmanager
from typing import Optional

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from db.database import close_db, init_db
from redis_consumer import BackendConsumer
from routers.agronomico import router as router_agronomico
from routers.cooperativa import router as router_cooperativa
from services.embedding_service import init_embedding_service
from services.gemini_service import GeminiService
from services.pipeline_service import PipelineService

LOG_LEVEL = settings.log_level.upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agrovoz.backend_api")

# ===========================================================
# Instancias globales (inicializadas en el lifespan)
# ===========================================================
gemini_service_instance:  Optional[GeminiService]  = None
pipeline_service_instance: Optional[PipelineService] = None
consumer_task: Optional[asyncio.Task] = None


# ===========================================================
# LIFESPAN: arranque y apagado ordenado
# ===========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager de lifespan para FastAPI.
    Código ANTES del yield = startup.
    Código DESPUÉS del yield = shutdown.
    """
    global gemini_service_instance, pipeline_service_instance, consumer_task

    logger.info("=" * 60)
    logger.info("🌾 AgroVoz — Backend API")
    logger.info("   Entorno: %s | Modelo: %s", settings.environment, settings.gemini_model)
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # STARTUP
    # ----------------------------------------------------------

    # 1. Inicializar pool de conexiones PostgreSQL
    logger.info("Inicializando conexión a PostgreSQL…")
    init_db(
        database_url=settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )

    # 2. Cargar modelo de embeddings (puede tardar en primer arranque)
    logger.info("Cargando modelo de embeddings '%s'…", settings.embedding_model)
    embedding_svc = init_embedding_service(
        model_name=settings.embedding_model,
        target_dim=settings.embedding_dim,
    )

    # 3. Inicializar cliente Gemini
    logger.info("Inicializando cliente Gemini '%s'…", settings.gemini_model)
    if not settings.gemini_api_key:
        logger.warning(
            "⚠️  GEMINI_API_KEY no configurada. El diagnóstico agronómico no funcionará."
        )
    gemini_service_instance = GeminiService(
        api_key=settings.gemini_api_key or "placeholder",
        model_name=settings.gemini_model,
        temperature=settings.gemini_temperature,
    )

    # 4. Inicializar servicio de pipeline
    pipeline_service_instance = PipelineService(
        gemini_service=gemini_service_instance,
        embedding_service=embedding_svc,
        rag_top_k=settings.rag_top_k,
        rag_min_similarity=settings.rag_min_similarity,
    )

    # 5. Arrancar BackendConsumer como tarea de fondo async
    consumer_name = os.getenv(
        "BACKEND_CONSUMER_NAME",
        f"backend_{socket.gethostname()}_{os.getpid()}"
    )
    consumer = BackendConsumer(
        redis_url=settings.redis_url,
        pipeline_service=pipeline_service_instance,
        consumer_name=consumer_name,
        group_name=settings.redis_group_backend,
        stream_texto=settings.redis_stream_texto,
        stream_respuesta=settings.redis_stream_respuesta,
    )

    consumer_task = asyncio.create_task(
        consumer.iniciar(),
        name="backend-consumer",
    )

    logger.info("✅ Backend API completamente inicializado. Listo para recibir peticiones.")

    yield  # ← FastAPI sirve peticiones durante este yield

    # ----------------------------------------------------------
    # SHUTDOWN
    # ----------------------------------------------------------
    logger.info("Iniciando apagado del backend…")

    # Detener consumer
    if consumer_task and not consumer_task.done():
        consumer.detener()
        try:
            await asyncio.wait_for(consumer_task, timeout=10.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()
            logger.warning("Consumer task cancelada por timeout.")

    # Cerrar pool de PostgreSQL
    await close_db()

    # Loggear métricas finales
    if gemini_service_instance:
        gemini_service_instance.log_metricas()
    if pipeline_service_instance:
        pipeline_service_instance.log_metricas()

    logger.info("🛑 Backend API apagado correctamente.")


# ===========================================================
# APLICACIÓN FASTAPI
# ===========================================================

app = FastAPI(
    title="AgroVoz — Backend API",
    description=(
        "API del asistente de voz para cooperativas agrícolas mexicanas. "
        "Procesa transcripciones de voz, ejecuta RAG agronómico con pgvector "
        "y gestiona trámites de cooperativa (entregas, solicitudes de insumos)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
    openapi_url="/openapi.json",
)

# CORS: ajustar origins en producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.es_desarrollo else ["https://app.cooperativa.mx"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(router_agronomico)
app.include_router(router_cooperativa)


# ===========================================================
# ENDPOINTS BASE
# ===========================================================

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "servicio": "AgroVoz Backend API",
        "version": "1.0.0",
        "estado": "activo",
        "docs": "/docs",
    }


@app.get("/health", tags=["Infraestructura"], summary="Health check del servicio")
async def health_check() -> JSONResponse:
    """
    Health check completo: verifica PostgreSQL, Redis y modelos de IA.
    Usado por Podman para healthchecks del contenedor.
    """
    checks: dict[str, str] = {}
    todo_ok = True

    # Check PostgreSQL
    try:
        from db.database import get_session
        from sqlalchemy import text
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        checks["postgresql"] = "ok"
    except Exception as e:
        checks["postgresql"] = f"error: {str(e)[:100]}"
        todo_ok = False

    # Check Redis
    try:
        r = redis.from_url(settings.redis_url, socket_connect_timeout=3)
        r.ping()
        r.close()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)[:100]}"
        todo_ok = False

    # Check modelos
    checks["gemini"]     = "ok" if gemini_service_instance else "no_inicializado"
    checks["embeddings"] = "ok" if pipeline_service_instance else "no_inicializado"
    checks["consumer"]   = (
        "ok" if consumer_task and not consumer_task.done()
        else "detenido"
    )

    status_code = 200 if todo_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "estado": "saludable" if todo_ok else "degradado",
            "checks": checks,
        },
    )


@app.get(
    "/metricas",
    tags=["Infraestructura"],
    summary="Métricas de operación del backend",
)
async def metricas() -> dict:
    """Retorna métricas de latencia y throughput de los servicios."""
    result: dict = {}

    if gemini_service_instance:
        result["gemini"] = {
            "total_llamadas":   gemini_service_instance._total_calls,
            "errores":          gemini_service_instance._errores,
            "latencia_prom_ms": round(gemini_service_instance.latencia_promedio_ms, 1),
        }

    if pipeline_service_instance:
        result["pipeline"] = {
            "procesados":       pipeline_service_instance._procesados,
            "errores":          pipeline_service_instance._errores,
            "flujo_agronomico": pipeline_service_instance._flujo_agronomico,
            "flujo_entrega":    pipeline_service_instance._flujo_entrega,
            "flujo_insumo":     pipeline_service_instance._flujo_insumo,
            "flujo_mixto":      pipeline_service_instance._flujo_mixto,
        }

    return result


# ===========================================================
# Entry point (desarrollo local sin contenedor)
# ===========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.es_desarrollo,
        log_level=settings.log_level.lower(),
    )
