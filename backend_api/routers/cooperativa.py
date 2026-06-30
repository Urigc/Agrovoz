"""
backend_api/routers/cooperativa.py
=====================================
Router FastAPI para operaciones de gestión de la cooperativa.

Endpoints:
  GET  /cooperativa/socios/{id_membresco}       → Info del socio
  GET  /cooperativa/socios/{id}/entregas        → Historial de entregas
  GET  /cooperativa/inventario                  → Lista de insumos con stock
  GET  /cooperativa/solicitudes                 → Solicitudes pendientes
  POST /cooperativa/solicitudes/{id}/aprobar    → Aprobar solicitud y descontar stock

Estos endpoints son usados principalmente por:
  - Operadores del almacén (web/móvil)
  - Scripts de auditoría
  - Tests de integración
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import text

from db.database import (
    EntregaCosechaRepository,
    InsumoRepository,
    SocioRepository,
    get_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cooperativa", tags=["Cooperativa"])


# ===========================================================
# Modelos de respuesta
# ===========================================================

class SocioResponse(BaseModel):
    id:                int
    nombre_completo:   str
    id_membresco:      str
    ubicacion_lote:    Optional[str]
    telefono_contacto: Optional[str]
    activo:            bool


class EntregaResponse(BaseModel):
    id:            int
    tipo_cultivo:  str
    cantidad_kg:   float
    calidad:       Optional[str]
    fecha_entrega: str
    origen:        str


class InsumoResponse(BaseModel):
    id:              int
    nombre_insumo:   str
    stock_actual:    float
    unidad_medida:   str
    punto_reorden:   Optional[float]
    precio_unitario: Optional[float]
    bajo_reorden:    bool   # True si stock <= punto_reorden


class SolicitudResponse(BaseModel):
    id:                  int
    socio_nombre:        str
    socio_membresco:     str
    insumo_nombre:       str
    cantidad_solicitada: float
    unidad_medida:       str
    estado:              str
    fecha_solicitud:     str
    origen:              str


# ===========================================================
# Endpoints — Socios
# ===========================================================

@router.get(
    "/socios/{id_membresco}",
    response_model=SocioResponse,
    summary="Buscar socio por ID de membresía",
)
async def obtener_socio(id_membresco: str) -> SocioResponse:
    """Busca un socio por su número de membresía (el que dicta al asistente de voz)."""
    async with get_session() as session:
        socio = await SocioRepository.buscar_por_membresco(session, id_membresco)

    if not socio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Socio con id_membresco='{id_membresco}' no encontrado.",
        )

    return SocioResponse(
        id=socio.id,
        nombre_completo=socio.nombre_completo,
        id_membresco=socio.id_membresco,
        ubicacion_lote=socio.ubicacion_lote,
        telefono_contacto=socio.telefono_contacto,
        activo=socio.activo,
    )


@router.get(
    "/socios/{id_membresco}/entregas",
    response_model=list[EntregaResponse],
    summary="Historial de entregas de un socio",
)
async def historial_entregas(
    id_membresco: str,
    limite: int = Query(default=10, ge=1, le=100),
) -> list[EntregaResponse]:
    """Retorna el historial de entregas de cosecha de un socio."""
    async with get_session() as session:
        socio = await SocioRepository.buscar_por_membresco(session, id_membresco)
        if not socio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Socio '{id_membresco}' no encontrado.",
            )

        entregas = await EntregaCosechaRepository.ultimas_entregas_socio(
            session, socio.id, limit=limite
        )

    return [
        EntregaResponse(
            id=int(e.get("id", 0)),
            tipo_cultivo=str(e.get("tipo_cultivo", "")),
            cantidad_kg=float(e.get("cantidad_kg", 0)),
            calidad=e.get("calidad"),
            fecha_entrega=str(e.get("fecha_entrega", "")),
            origen=str(e.get("origen", "voz")),
        )
        for e in entregas
    ]


# ===========================================================
# Endpoints — Inventario
# ===========================================================

@router.get(
    "/inventario",
    response_model=list[InsumoResponse],
    summary="Lista de insumos del almacén con stock actual",
)
async def listar_inventario(
    solo_bajo_reorden: bool = Query(
        default=False,
        description="Si True, retorna solo insumos con stock bajo el punto de reorden."
    ),
    buscar: Optional[str] = Query(
        default=None,
        description="Filtrar por nombre de insumo (búsqueda parcial)."
    ),
) -> list[InsumoResponse]:
    """
    Lista el inventario completo de insumos con indicador de stock bajo.
    Útil para que los operadores del almacén identifiquen qué reponer.
    """
    async with get_session() as session:
        query_base = """
            SELECT
                id, nombre_insumo, stock_actual, unidad_medida,
                punto_reorden, precio_unitario,
                CASE
                    WHEN punto_reorden IS NOT NULL AND stock_actual <= punto_reorden
                    THEN TRUE ELSE FALSE
                END AS bajo_reorden
            FROM inventario_insumos
            WHERE activo = TRUE
        """
        params: dict = {}

        if buscar:
            query_base += " AND nombre_insumo ILIKE :buscar"
            params["buscar"] = f"%{buscar}%"

        if solo_bajo_reorden:
            query_base += " AND punto_reorden IS NOT NULL AND stock_actual <= punto_reorden"

        query_base += " ORDER BY nombre_insumo ASC"

        result = await session.execute(text(query_base), params)
        insumos = [dict(row._mapping) for row in result]

    return [
        InsumoResponse(
            id=int(i["id"]),
            nombre_insumo=str(i["nombre_insumo"]),
            stock_actual=float(i["stock_actual"]),
            unidad_medida=str(i["unidad_medida"]),
            punto_reorden=float(i["punto_reorden"]) if i.get("punto_reorden") else None,
            precio_unitario=float(i["precio_unitario"]) if i.get("precio_unitario") else None,
            bajo_reorden=bool(i.get("bajo_reorden", False)),
        )
        for i in insumos
    ]


# ===========================================================
# Endpoints — Solicitudes
# ===========================================================

@router.get(
    "/solicitudes",
    response_model=list[SolicitudResponse],
    summary="Lista de solicitudes de insumos",
)
async def listar_solicitudes(
    estado: Optional[str] = Query(
        default="Pendiente",
        description="Filtrar por estado: Pendiente, Aprobado, Entregado, Rechazado",
    ),
    limite: int = Query(default=50, ge=1, le=500),
) -> list[SolicitudResponse]:
    """
    Lista solicitudes de insumos, por defecto las pendientes de aprobación.
    Usada por el operador del almacén para gestionar el flujo de solicitudes.
    """
    async with get_session() as session:
        query = """
            SELECT
                si.id,
                s.nombre_completo AS socio_nombre,
                s.id_membresco   AS socio_membresco,
                ii.nombre_insumo,
                ii.unidad_medida,
                si.cantidad_solicitada,
                si.estado,
                si.fecha_solicitud,
                si.origen
            FROM solicitudes_insumos si
            JOIN socios s             ON s.id = si.socio_id
            JOIN inventario_insumos ii ON ii.id = si.insumo_id
            WHERE (:estado IS NULL OR si.estado = :estado)
            ORDER BY si.fecha_solicitud DESC
            LIMIT :limite
        """
        result = await session.execute(
            text(query),
            {"estado": estado, "limite": limite},
        )
        solicitudes = [dict(row._mapping) for row in result]

    return [
        SolicitudResponse(
            id=int(s["id"]),
            socio_nombre=str(s["socio_nombre"]),
            socio_membresco=str(s["socio_membresco"]),
            insumo_nombre=str(s["nombre_insumo"]),
            cantidad_solicitada=float(s["cantidad_solicitada"]),
            unidad_medida=str(s["unidad_medida"]),
            estado=str(s["estado"]),
            fecha_solicitud=str(s["fecha_solicitud"]),
            origen=str(s["origen"]),
        )
        for s in solicitudes
    ]


@router.post(
    "/solicitudes/{solicitud_id}/aprobar",
    summary="Aprobar solicitud de insumo y descontar stock",
    status_code=status.HTTP_200_OK,
)
async def aprobar_solicitud(solicitud_id: int) -> dict:
    """
    Aprueba una solicitud de insumo pendiente:
    1. Cambia estado de 'Pendiente' a 'Aprobado'
    2. Descuenta la cantidad del stock de inventario

    Esta acción es atómica: si el descuento de stock falla, el estado no cambia.
    """
    async with get_session() as session:
        # Obtener datos de la solicitud
        result = await session.execute(
            text("""
                SELECT si.id, si.insumo_id, si.cantidad_solicitada, si.estado,
                       ii.nombre_insumo, ii.stock_actual, ii.unidad_medida
                FROM solicitudes_insumos si
                JOIN inventario_insumos ii ON ii.id = si.insumo_id
                WHERE si.id = :id
            """),
            {"id": solicitud_id},
        )
        row = result.mappings().first()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Solicitud {solicitud_id} no encontrada.",
            )

        if row["estado"] != "Pendiente":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Solicitud {solicitud_id} ya tiene estado '{row['estado']}'. Solo se pueden aprobar solicitudes 'Pendiente'.",
            )

        # Verificar y descontar stock
        tiene_stock = await InsumoRepository.descontar_stock(
            session, row["insumo_id"], float(row["cantidad_solicitada"])
        )

        if not tiene_stock:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Stock insuficiente para aprobar. "
                    f"Stock actual de '{row['nombre_insumo']}': {row['stock_actual']} {row['unidad_medida']}. "
                    f"Solicitado: {row['cantidad_solicitada']} {row['unidad_medida']}."
                ),
            )

        # Actualizar estado a 'Aprobado'
        await session.execute(
            text("""
                UPDATE solicitudes_insumos
                SET estado = 'Aprobado', fecha_actualizacion = NOW()
                WHERE id = :id
            """),
            {"id": solicitud_id},
        )

        logger.info(
            "Solicitud %d aprobada: '%s' × %.1f %s",
            solicitud_id,
            row["nombre_insumo"],
            float(row["cantidad_solicitada"]),
            row["unidad_medida"],
        )

    return {
        "mensaje": f"Solicitud {solicitud_id} aprobada correctamente.",
        "insumo":  row["nombre_insumo"],
        "cantidad_aprobada": float(row["cantidad_solicitada"]),
        "unidad":  row["unidad_medida"],
    }
