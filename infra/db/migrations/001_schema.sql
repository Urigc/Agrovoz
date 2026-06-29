-- ============================================================
-- AgroVoz — Migración 001: Schema Completo
-- Archivo: infra/db/migrations/001_schema.sql
--
-- Este script es idempotente (usa IF NOT EXISTS / IF EXISTS).
-- Se ejecuta automáticamente en el primer arranque del contenedor
-- PostgreSQL gracias al mecanismo de /docker-entrypoint-initdb.d/.
--
-- Orden de ejecución:
--   1. Extensiones
--   2. Tablas relacionales (cooperativa)
--   3. Tabla vectorial (RAG agronómico)
--   4. Índices (HNSW para pgvector, B-tree para queries frecuentes)
--   5. Trigger de auditoría de stock
-- ============================================================

-- ============================================================
-- 1. EXTENSIONES
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector: embeddings y búsqueda ANN
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- Búsqueda fuzzy en nombres de insumos
CREATE EXTENSION IF NOT EXISTS btree_gin;    -- Índices GIN en columnas escalares

-- ============================================================
-- 2. TABLAS RELACIONALES — Gestión de Cooperativa
-- ============================================================

-- ------------------------------------------------------------
-- Socios / Agricultores
-- La columna id_membresco es la que el agricultor dicta por voz
-- (ej: "soy el socio cuatrocientos dos"). Debe ser fácil de decir.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS socios (
    id               SERIAL PRIMARY KEY,
    nombre_completo  VARCHAR(150) NOT NULL,
    id_membresco     VARCHAR(50)  UNIQUE NOT NULL,   -- "402", "COOP-001", etc.
    ubicacion_lote   VARCHAR(255),                    -- "Parcela norte, km 12"
    telefono_contacto VARCHAR(20),
    activo           BOOLEAN      NOT NULL DEFAULT TRUE,
    fecha_registro   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE socios IS 'Agricultores miembros de la cooperativa. id_membresco es el identificador dictado por voz.';
COMMENT ON COLUMN socios.id_membresco IS 'Código corto y memorable que el agricultor dicta al asistente. Ej: 402, A-001.';

-- ------------------------------------------------------------
-- Inventario de Insumos
-- Fertilizantes, agroquímicos, herramientas, semillas, etc.
-- El campo nombre_insumo se usa para matching fuzzy desde texto dictado.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inventario_insumos (
    id              SERIAL PRIMARY KEY,
    nombre_insumo   VARCHAR(100) NOT NULL,           -- "Urea 46%", "Herbicida Roundup"
    descripcion     TEXT,
    stock_actual    DECIMAL(12, 3) NOT NULL DEFAULT 0,
    unidad_medida   VARCHAR(20)  NOT NULL             -- 'kg', 'litros', 'unidades', 'sacos'
                    CHECK (unidad_medida IN ('kg', 'litros', 'unidades', 'sacos', 'cajas')),
    punto_reorden   DECIMAL(12, 3),                  -- Alerta cuando stock cae aquí
    precio_unitario DECIMAL(10, 2),                  -- Precio por unidad (referencia)
    activo          BOOLEAN      NOT NULL DEFAULT TRUE,
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE inventario_insumos IS 'Catálogo de insumos del almacén de la cooperativa con control de stock.';
COMMENT ON COLUMN inventario_insumos.punto_reorden IS 'Si stock_actual cae por debajo de este valor, el sistema puede alertar.';

-- ------------------------------------------------------------
-- Entregas de Cosecha
-- Reemplaza el "papel de recepción de grano" físico.
-- Cada registro = un agricultor entrega X kg de un cultivo.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entregas_cosecha (
    id             SERIAL PRIMARY KEY,
    socio_id       INT          NOT NULL REFERENCES socios(id) ON DELETE RESTRICT,
    tipo_cultivo   VARCHAR(50)  NOT NULL               -- 'maiz', 'frijol', 'cafe', 'trigo'
                   CHECK (tipo_cultivo IN ('maiz', 'frijol', 'cafe', 'trigo', 'arroz', 'sorgo', 'otro')),
    cantidad_kg    DECIMAL(10, 3) NOT NULL
                   CHECK (cantidad_kg > 0),
    calidad        VARCHAR(20)                          -- 'Premium', 'Estándar', 'Rechazado'
                   CHECK (calidad IN ('Premium', 'Estándar', 'Rechazado')),
    precio_por_kg  DECIMAL(8, 2),                      -- Precio pactado al momento de entrega
    notas          TEXT,                               -- Observaciones del operador
    fecha_entrega  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    -- Auditoría: qué canal generó este registro
    origen         VARCHAR(20)  NOT NULL DEFAULT 'voz'
                   CHECK (origen IN ('voz', 'manual', 'importacion'))
);

COMMENT ON TABLE entregas_cosecha IS 'Reemplaza el formulario físico de recepción de cosecha. Registrable por voz.';
COMMENT ON COLUMN entregas_cosecha.origen IS 'Canal de ingreso: voz (AgroVoz), manual (operador), importacion (legacy).';

-- ------------------------------------------------------------
-- Solicitudes de Insumos
-- Reemplaza el "papel de pedido al almacén".
-- Un socio pide N unidades de un insumo; el almacén aprueba/entrega.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS solicitudes_insumos (
    id                   SERIAL PRIMARY KEY,
    socio_id             INT          NOT NULL REFERENCES socios(id) ON DELETE RESTRICT,
    insumo_id            INT          NOT NULL REFERENCES inventario_insumos(id) ON DELETE RESTRICT,
    cantidad_solicitada  DECIMAL(12, 3) NOT NULL
                         CHECK (cantidad_solicitada > 0),
    estado               VARCHAR(20)  NOT NULL DEFAULT 'Pendiente'
                         CHECK (estado IN ('Pendiente', 'Aprobado', 'Entregado', 'Rechazado', 'Cancelado')),
    notas_operador       TEXT,
    fecha_solicitud      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    fecha_actualizacion  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    origen               VARCHAR(20)  NOT NULL DEFAULT 'voz'
                         CHECK (origen IN ('voz', 'manual'))
);

COMMENT ON TABLE solicitudes_insumos IS 'Pedidos de insumos al almacén. El flujo: Pendiente → Aprobado → Entregado.';

-- ============================================================
-- 3. TABLA VECTORIAL — RAG Agronómico
-- ============================================================

-- ------------------------------------------------------------
-- Base de Conocimiento Agronómico
-- Fragmentos de manuales, guías de plagas, fichas de cultivos.
-- Cada fila = un chunk de texto con su embedding para búsqueda ANN.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conocimiento_agronomico (
    id                   SERIAL PRIMARY KEY,
    titulo_documento     VARCHAR(255) NOT NULL,        -- "Manual de Plagas del Maíz v3"
    tipo_cultivo_aplica  VARCHAR(50),                  -- 'maiz', 'frijol', NULL = general
    seccion              VARCHAR(100),                 -- "Capítulo 4: Hongos Foliares"
    fragmento_texto      TEXT         NOT NULL,        -- El chunk de texto real
    metadata             JSONB        NOT NULL DEFAULT '{}',
    -- JSONB metadata puede incluir: fuente, año, autor, página, idioma, etc.
    embedding            vector(1536) NOT NULL,        -- Vector del fragmento (text-embedding-ada-002 o similar)
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE conocimiento_agronomico IS 'Base RAG: fragmentos de manuales agronómicos con embeddings para búsqueda semántica.';
COMMENT ON COLUMN conocimiento_agronomico.embedding IS 'Vector de 1536 dims compatible con text-embedding-ada-002. Indexado con HNSW coseno.';
COMMENT ON COLUMN conocimiento_agronomico.metadata IS 'JSON libre: {"fuente": "CIMMYT", "año": 2022, "pagina": 47, "idioma": "es"}';

-- ============================================================
-- 4. ÍNDICES
-- ============================================================

-- B-tree estándar para queries relacionales frecuentes
CREATE INDEX IF NOT EXISTS idx_socios_id_membresco
    ON socios (id_membresco);

CREATE INDEX IF NOT EXISTS idx_entregas_socio_fecha
    ON entregas_cosecha (socio_id, fecha_entrega DESC);

CREATE INDEX IF NOT EXISTS idx_entregas_cultivo_fecha
    ON entregas_cosecha (tipo_cultivo, fecha_entrega DESC);

CREATE INDEX IF NOT EXISTS idx_solicitudes_socio_estado
    ON solicitudes_insumos (socio_id, estado, fecha_solicitud DESC);

CREATE INDEX IF NOT EXISTS idx_solicitudes_estado
    ON solicitudes_insumos (estado, fecha_solicitud DESC);

-- GIN trigram para búsqueda fuzzy de insumos por nombre dictado
-- Permite: "SELECT * FROM inventario_insumos WHERE nombre_insumo % 'urea'"
CREATE INDEX IF NOT EXISTS idx_insumos_nombre_trgm
    ON inventario_insumos USING gin (nombre_insumo gin_trgm_ops);

-- GIN para consultas sobre metadata JSONB
CREATE INDEX IF NOT EXISTS idx_conocimiento_metadata
    ON conocimiento_agronomico USING gin (metadata);

CREATE INDEX IF NOT EXISTS idx_conocimiento_cultivo
    ON conocimiento_agronomico (tipo_cultivo_aplica);

-- HNSW para búsqueda de similitud coseno — el corazón del sistema RAG
-- m=16, ef_construction=64: buen equilibrio velocidad/precisión para 1M vectores
-- Objetivo: < 50ms de latencia en búsqueda con 1M de registros
CREATE INDEX IF NOT EXISTS idx_conocimiento_embedding_hnsw
    ON conocimiento_agronomico
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- 5. FUNCIÓN Y TRIGGER: Actualizar updated_at automáticamente
-- ============================================================

CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_insumos_updated_at
    BEFORE UPDATE ON inventario_insumos
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

CREATE OR REPLACE FUNCTION fn_set_solicitud_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fecha_actualizacion = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_solicitudes_updated_at
    BEFORE UPDATE ON solicitudes_insumos
    FOR EACH ROW EXECUTE FUNCTION fn_set_solicitud_updated_at();

-- ============================================================
-- 6. FUNCIÓN UTILITARIA: Búsqueda RAG por similitud coseno
-- Usada por el backend_api para el flujo de diagnóstico agronómico
-- ============================================================

CREATE OR REPLACE FUNCTION buscar_conocimiento_agronomico(
    query_embedding  vector(1536),
    cultivo_filtro   VARCHAR(50) DEFAULT NULL,
    top_k            INT         DEFAULT 3
)
RETURNS TABLE (
    id               INT,
    titulo_documento VARCHAR(255),
    tipo_cultivo_aplica VARCHAR(50),
    fragmento_texto  TEXT,
    metadata         JSONB,
    similitud        FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        k.id,
        k.titulo_documento,
        k.tipo_cultivo_aplica,
        k.fragmento_texto,
        k.metadata,
        1 - (k.embedding <=> query_embedding) AS similitud
    FROM conocimiento_agronomico k
    WHERE
        (cultivo_filtro IS NULL OR k.tipo_cultivo_aplica = cultivo_filtro)
    ORDER BY
        k.embedding <=> query_embedding   -- operador coseno de pgvector
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql STABLE PARALLEL SAFE;

COMMENT ON FUNCTION buscar_conocimiento_agronomico IS
'Búsqueda RAG: recupera los top_k fragmentos más similares al embedding de la consulta. Usada por backend_api.';

-- ============================================================
-- Log de migración exitosa
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE 'AgroVoz — Migración 001 aplicada correctamente. % tablas creadas.', 
        (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public');
END $$;
