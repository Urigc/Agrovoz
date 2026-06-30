"""
backend_api/db/database.py
============================
Capa de acceso a datos async para AgroVoz.

Contiene:
  - Engine async de SQLAlchemy con asyncpg
  - Modelos ORM mapeados a las tablas del schema SQL
  - Repositorios: funciones de lectura/escritura para cada tabla
  - Función de búsqueda RAG usando pgvector

Diseño:
  - Todo es async/await — un solo proceso puede manejar múltiples
    agricultores simultáneos sin bloqueos.
  - Sin ORM pesado: usamos Core SQL de SQLAlchemy para queries críticos
    (entregas, solicitudes) y la función SQL buscar_conocimiento_agronomico()
    para RAG, evitando N+1 y manteniendo latencia < 50ms.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, Column, DateTime, DECIMAL, ForeignKey,
    Integer, String, Text, text, JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

logger = logging.getLogger(__name__)

# ===========================================================
# ENGINE Y SESSION FACTORY
# ===========================================================

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


def init_db(database_url: str, pool_size: int = 10, max_overflow: int = 5) -> None:
    """
    Inicializa el engine async y la session factory.
    Llamar una vez al arrancar la aplicación FastAPI.
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=30,
        pool_recycle=1800,    # Reciclar conexiones cada 30min (evita conexiones muertas)
        pool_pre_ping=True,   # Verificar conexión antes de usar (resiliencia)
        echo=False,           # True para debug SQL (muy verboso en producción)
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,   # Evita lazy loading tras commit
        autoflush=True,
        autocommit=False,
    )

    logger.info("Database engine inicializado: pool_size=%d", pool_size)


async def close_db() -> None:
    """Cierra el pool de conexiones al apagar la aplicación."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database engine cerrado.")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager para obtener una sesión de base de datos.
    Hace rollback automático en caso de excepción.

    Uso:
        async with get_session() as session:
            result = await session.execute(select(Socio))
    """
    if not _session_factory:
        raise RuntimeError("Base de datos no inicializada. Llamar init_db() primero.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ===========================================================
# MODELOS ORM (mapeados al schema SQL del migration 001)
# ===========================================================

class Base(DeclarativeBase):
    pass


class Socio(Base):
    __tablename__ = "socios"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    nombre_completo   = Column(String(150), nullable=False)
    id_membresco      = Column(String(50),  unique=True, nullable=False)
    ubicacion_lote    = Column(String(255))
    telefono_contacto = Column(String(20))
    activo            = Column(Boolean, default=True, nullable=False)
    fecha_registro    = Column(DateTime(timezone=True), default=datetime.utcnow)

    entregas   = relationship("EntregaCosecha",    back_populates="socio")
    solicitudes = relationship("SolicitudInsumo",  back_populates="socio")


class InventarioInsumo(Base):
    __tablename__ = "inventario_insumos"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    nombre_insumo   = Column(String(100), nullable=False)
    descripcion     = Column(Text)
    stock_actual    = Column(DECIMAL(12, 3), nullable=False, default=0)
    unidad_medida   = Column(String(20), nullable=False)
    punto_reorden   = Column(DECIMAL(12, 3))
    precio_unitario = Column(DECIMAL(10, 2))
    activo          = Column(Boolean, default=True, nullable=False)

    solicitudes = relationship("SolicitudInsumo", back_populates="insumo")


class EntregaCosecha(Base):
    __tablename__ = "entregas_cosecha"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    socio_id       = Column(Integer, ForeignKey("socios.id"), nullable=False)
    tipo_cultivo   = Column(String(50), nullable=False)
    cantidad_kg    = Column(DECIMAL(10, 3), nullable=False)
    calidad        = Column(String(20))
    precio_por_kg  = Column(DECIMAL(8, 2))
    notas          = Column(Text)
    fecha_entrega  = Column(DateTime(timezone=True), default=datetime.utcnow)
    origen         = Column(String(20), default="voz", nullable=False)

    socio = relationship("Socio", back_populates="entregas")


class SolicitudInsumo(Base):
    __tablename__ = "solicitudes_insumos"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    socio_id             = Column(Integer, ForeignKey("socios.id"),             nullable=False)
    insumo_id            = Column(Integer, ForeignKey("inventario_insumos.id"), nullable=False)
    cantidad_solicitada  = Column(DECIMAL(12, 3), nullable=False)
    estado               = Column(String(20), default="Pendiente", nullable=False)
    notas_operador       = Column(Text)
    fecha_solicitud      = Column(DateTime(timezone=True), default=datetime.utcnow)
    fecha_actualizacion  = Column(DateTime(timezone=True), default=datetime.utcnow)
    origen               = Column(String(20), default="voz", nullable=False)

    socio  = relationship("Socio",             back_populates="solicitudes")
    insumo = relationship("InventarioInsumo",  back_populates="solicitudes")


class ConocimientoAgronomico(Base):
    __tablename__ = "conocimiento_agronomico"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    titulo_documento     = Column(String(255), nullable=False)
    tipo_cultivo_aplica  = Column(String(50))
    seccion              = Column(String(100))
    fragmento_texto      = Column(Text, nullable=False)
    # Nota: 'metadata' es reservado por SQLAlchemy ORM → usamos 'meta' como atributo Python
    # pero el nombre de columna en la DB sigue siendo 'metadata'
    meta                 = Column("metadata", JSONB, default=dict)
    embedding            = Column(Vector(1536), nullable=False)
    created_at           = Column(DateTime(timezone=True), default=datetime.utcnow)


# ===========================================================
# REPOSITORIOS — Funciones de acceso a datos
# ===========================================================

class SocioRepository:
    """Operaciones de lectura sobre la tabla socios."""

    @staticmethod
    async def buscar_por_membresco(
        session: AsyncSession, id_membresco: str
    ) -> Optional[Socio]:
        """Busca un socio por su ID de membresía (el que dicta el agricultor)."""
        result = await session.execute(
            text("SELECT * FROM socios WHERE id_membresco = :id AND activo = TRUE"),
            {"id": id_membresco.strip()},
        )
        row = result.mappings().first()
        if row:
            s = Socio()
            for k, v in row.items():
                setattr(s, k, v)
            return s
        return None


class EntregaCosechaRepository:
    """Operaciones sobre entregas de cosecha."""

    @staticmethod
    async def registrar(
        session: AsyncSession,
        socio_id:     int,
        tipo_cultivo: str,
        cantidad_kg:  float,
        calidad:      Optional[str] = None,
        precio_por_kg: Optional[float] = None,
        notas:        Optional[str] = None,
    ) -> int:
        """
        Inserta una nueva entrega de cosecha.
        Retorna el ID del registro creado.
        """
        result = await session.execute(
            text("""
                INSERT INTO entregas_cosecha
                    (socio_id, tipo_cultivo, cantidad_kg, calidad, precio_por_kg, notas, origen)
                VALUES
                    (:socio_id, :tipo_cultivo, :cantidad_kg, :calidad, :precio_por_kg, :notas, 'voz')
                RETURNING id
            """),
            {
                "socio_id":     socio_id,
                "tipo_cultivo": tipo_cultivo.lower().strip(),
                "cantidad_kg":  cantidad_kg,
                "calidad":      calidad,
                "precio_por_kg": precio_por_kg,
                "notas":        notas,
            },
        )
        row = result.first()
        return row[0] if row else -1

    @staticmethod
    async def ultimas_entregas_socio(
        session: AsyncSession, socio_id: int, limit: int = 5
    ) -> list[dict]:
        """Retorna las últimas N entregas de un socio (para confirmación por voz)."""
        result = await session.execute(
            text("""
                SELECT tipo_cultivo, cantidad_kg, calidad, fecha_entrega
                FROM entregas_cosecha
                WHERE socio_id = :socio_id
                ORDER BY fecha_entrega DESC
                LIMIT :limit
            """),
            {"socio_id": socio_id, "limit": limit},
        )
        return [dict(row._mapping) for row in result]


class InsumoRepository:
    """Operaciones sobre inventario de insumos."""

    @staticmethod
    async def buscar_por_nombre_fuzzy(
        session: AsyncSession, nombre: str, limit: int = 3
    ) -> list[dict]:
        """
        Búsqueda fuzzy por nombre (pg_trgm) para matchear nombres dictados informalmente.
        Ej: "urea" → "Urea 46%", "roundup" → "Herbicida Glifosato 36%"
        """
        result = await session.execute(
            text("""
                SELECT id, nombre_insumo, stock_actual, unidad_medida, punto_reorden
                FROM inventario_insumos
                WHERE activo = TRUE
                  AND (
                    nombre_insumo ILIKE :patron
                    OR similarity(nombre_insumo, :nombre) > 0.2
                  )
                ORDER BY similarity(nombre_insumo, :nombre) DESC
                LIMIT :limit
            """),
            {"nombre": nombre.strip(), "patron": f"%{nombre.strip()}%", "limit": limit},
        )
        return [dict(row._mapping) for row in result]

    @staticmethod
    async def verificar_stock(
        session: AsyncSession, insumo_id: int, cantidad_requerida: float
    ) -> tuple[bool, float]:
        """
        Verifica si hay suficiente stock de un insumo.
        Retorna (tiene_stock: bool, stock_actual: float).
        """
        result = await session.execute(
            text("SELECT stock_actual FROM inventario_insumos WHERE id = :id AND activo = TRUE"),
            {"id": insumo_id},
        )
        row = result.first()
        if not row:
            return False, 0.0
        stock = float(row[0])
        return stock >= cantidad_requerida, stock

    @staticmethod
    async def descontar_stock(
        session: AsyncSession, insumo_id: int, cantidad: float
    ) -> bool:
        """
        Descuenta stock al aprobar/entregar una solicitud.
        Retorna True si el descuento fue exitoso (stock suficiente).
        """
        result = await session.execute(
            text("""
                UPDATE inventario_insumos
                SET stock_actual = stock_actual - :cantidad
                WHERE id = :id
                  AND stock_actual >= :cantidad
                  AND activo = TRUE
                RETURNING id, stock_actual
            """),
            {"id": insumo_id, "cantidad": cantidad},
        )
        return result.first() is not None


class SolicitudInsumoRepository:
    """Operaciones sobre solicitudes de insumos."""

    @staticmethod
    async def crear(
        session: AsyncSession,
        socio_id:            int,
        insumo_id:           int,
        cantidad_solicitada: float,
    ) -> int:
        """Crea una solicitud de insumo con estado 'Pendiente'. Retorna el ID."""
        result = await session.execute(
            text("""
                INSERT INTO solicitudes_insumos
                    (socio_id, insumo_id, cantidad_solicitada, estado, origen)
                VALUES
                    (:socio_id, :insumo_id, :cantidad, 'Pendiente', 'voz')
                RETURNING id
            """),
            {
                "socio_id":  socio_id,
                "insumo_id": insumo_id,
                "cantidad":  cantidad_solicitada,
            },
        )
        row = result.first()
        return row[0] if row else -1


class RAGRepository:
    """Búsqueda semántica en la base de conocimiento agronómico."""

    @staticmethod
    async def buscar_fragmentos(
        session:        AsyncSession,
        embedding:      list[float],
        top_k:          int = 3,
        cultivo_filtro: Optional[str] = None,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """
        Busca los fragmentos de conocimiento agronómico más similares al embedding
        de la consulta del agricultor. Usa el índice HNSW de pgvector.

        Args:
            embedding:      Vector de la consulta (1536 dims).
            top_k:          Número de fragmentos a recuperar.
            cultivo_filtro: Si se conoce el cultivo, filtra para mayor precisión.
            min_similarity: Umbral mínimo de similitud coseno (0.3 = relevante).

        Returns:
            Lista de dicts con fragmento_texto, titulo_documento, similitud.
        """
        # Convertir embedding Python list → formato vector de pgvector
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        # Usar la función SQL definida en la migración 001
        result = await session.execute(
            text("""
                SELECT
                    id,
                    titulo_documento,
                    tipo_cultivo_aplica,
                    fragmento_texto,
                    metadata,
                    similitud
                FROM buscar_conocimiento_agronomico(
                    :embedding::vector,
                    :cultivo_filtro,
                    :top_k
                )
                WHERE similitud >= :min_similarity
                ORDER BY similitud DESC
            """),
            {
                "embedding":      embedding_str,
                "cultivo_filtro": cultivo_filtro,
                "top_k":          top_k,
                "min_similarity": min_similarity,
            },
        )
        return [dict(row._mapping) for row in result]
