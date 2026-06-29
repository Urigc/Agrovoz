-- ============================================================
-- AgroVoz — Seed 001: Datos Iniciales Realistas
-- Archivo: infra/db/seeds/001_datos_iniciales.sql
--
-- Ejecutar DESPUÉS de la migración 001_schema.sql.
-- Contiene:
--   - 10 socios de ejemplo (agricultores reales ficticios)
--   - 20 insumos del inventario (fertilizantes, agroquímicos comunes en MX)
--   - 5 fragmentos de conocimiento agronómico de muestra
--     (sin embeddings reales — los vectores se cargan via script Python)
--
-- NOTA: Los embeddings en esta seed son vectores cero (placeholder).
-- El script scripts/cargar_conocimiento.py los reemplaza con vectores reales.
-- ============================================================

-- ============================================================
-- SOCIOS (Agricultores de la cooperativa)
-- ============================================================
INSERT INTO socios (nombre_completo, id_membresco, ubicacion_lote, telefono_contacto)
VALUES
    ('Juan Pérez Hernández',      '001',  'Parcela Norte, Lote 1-A, Km 12 Carr. Toluca',   '7221001001'),
    ('María López Ramírez',       '002',  'Parcela Sur, Lote 2-B, Ejido San Miguel',        '7221001002'),
    ('Carlos Gutiérrez Morales',  '100',  'Rancho El Sabinal, Municipio de Joquicingo',     '7221001003'),
    ('Rosa Elena Juárez Castro',  '101',  'Parcela Oriente, Comunidad La Lagunilla',        '7221001004'),
    ('Antonio Flores Mendoza',    '200',  'Lote 5, Unidad de Riego Núm. 3',                 '7221001005'),
    ('Guadalupe Martínez Torres', '201',  'Parcela Centro, km 8 Carr. Ixtapan de la Sal',  '7221001006'),
    ('Pedro Sánchez Ríos',        '300',  'Rancho Las Truchas, predio colindante con río',  '7221001007'),
    ('Elena Ruiz Vásquez',        '301',  'Ejido El Carmen, parcela 14',                    '7221001008'),
    ('Rodrigo Torres Salinas',    '400',  'Terreno Cerro del Xitle, 1.5 ha',               '7221001009'),
    ('Sofía Ramírez Luna',        '402',  'Parcela familiar Rancho La Palma, 2 ha',         '7221001010')
ON CONFLICT (id_membresco) DO NOTHING;

-- ============================================================
-- INVENTARIO DE INSUMOS
-- Basado en insumos comunes para maíz, frijol y café en México central
-- ============================================================
INSERT INTO inventario_insumos (nombre_insumo, descripcion, stock_actual, unidad_medida, punto_reorden, precio_unitario)
VALUES
    -- Fertilizantes nitrogenados
    ('Urea 46%',
     'Fertilizante nitrogenado sólido. Fórmula N-P-K: 46-0-0. Uso: aplicación al suelo o foliar.',
     2500.000, 'kg', 500.000, 12.50),

    ('Sulfato de Amonio 21%',
     'Fertilizante nitrogenado con azufre. Fórmula: 21-0-0-24S. Ideal para suelos alcalinos.',
     1800.000, 'kg', 300.000, 9.80),

    ('Nitrato de Amonio 33%',
     'Alta concentración de nitrógeno. Uso en fertirrigación y aplicación directa.',
     1200.000, 'kg', 200.000, 15.30),

    -- Fertilizantes fosfatados
    ('Superfosfato Triple 46%',
     'Fósforo de alta concentración: 0-46-0. Estimula desarrollo radicular y floración.',
     900.000, 'kg', 150.000, 18.90),

    ('DAP (Fosfato Diamónico)',
     'Arranque de cultivos. Fórmula: 18-46-0. Muy usado en siembra de maíz y frijol.',
     1500.000, 'kg', 250.000, 22.00),

    -- Fertilizantes compuestos
    ('Fertilizante 17-17-17',
     'Fórmula balanceada NPK. Apto para mantenimiento general de cultivos.',
     2000.000, 'kg', 400.000, 19.50),

    ('Fertilizante 20-10-10 + micronutrientes',
     'Alto nitrógeno con micronutrientes (Zn, B, Fe). Ideal para maíz en etapa vegetativa.',
     1100.000, 'kg', 200.000, 24.00),

    -- Agroquímicos: herbicidas
    ('Herbicida Glifosato 36%',
     'Herbicida sistémico no selectivo. Control de maleza de hoja ancha y gramíneas. Uso pre-siembra.',
     450.000, 'litros', 50.000, 85.00),

    ('Herbicida Atrazina 90% WP',
     'Control de maleza en maíz. Selectivo. Aplicar en pre-emergencia o post-emergencia temprana.',
     300.000, 'kg', 40.000, 120.00),

    ('Herbicida 2,4-D Amina 72%',
     'Control de maleza de hoja ancha en cereales. Muy económico y efectivo.',
     380.000, 'litros', 60.000, 65.00),

    -- Agroquímicos: fungicidas
    ('Fungicida Mancozeb 80% WP',
     'Protección preventiva contra roya, tizón y mildiu. Amplio espectro en frijol y café.',
     280.000, 'kg', 30.000, 145.00),

    ('Fungicida Tebuconazol 25%',
     'Control curativo de roya del café y enfermedades foliares. Acción sistémica.',
     120.000, 'litros', 20.000, 380.00),

    ('Fungicida Carbendazim 50%',
     'Control de antracnosis, pudrición de raíz y enfermedades fungosas del suelo.',
     95.000, 'litros', 15.000, 290.00),

    -- Agroquímicos: insecticidas
    ('Insecticida Clorpirifos 480 EC',
     'Control de gusano cogollero, pulgones y plagas del suelo. Uso restringido, requiere EPP.',
     200.000, 'litros', 25.000, 195.00),

    ('Insecticida Cipermetrina 20%',
     'Piretroide de amplio espectro. Plagas en maíz y frijol. Alta eficacia, baja dosis.',
     180.000, 'litros', 20.000, 160.00),

    ('Insecticida Bacillus thuringiensis (Bt)',
     'Biopesticida para gusano cogollero. Bajo impacto ambiental. Certificado orgánico.',
     350.000, 'kg', 50.000, 220.00),

    -- Semillas
    ('Semilla de Maíz Híbrido H-520',
     'Híbrido de alto rendimiento. Ciclo: 120 días. Tolerante a sequía. 10 kg/bolsa.',
     180.000, 'sacos', 20.000, 850.00),

    ('Semilla de Frijol Negro Jamapa',
     'Variedad mejorada. Ciclo: 65 días. Alta proteína. Adaptada a clima templado.',
     90.000, 'sacos', 10.000, 420.00),

    -- Cal y correctores
    ('Cal Agrícola (Carbonato de Calcio)',
     'Corrector de pH ácido. Mejora estructura del suelo. Aplicar 3-6 meses antes de siembra.',
     8000.000, 'kg', 1000.000, 3.20),

    -- Equipo menor
    ('Bomba de mochila 20L (aspersora)',
     'Bomba manual de presión para aplicación de agroquímicos. Capacidad 20 litros.',
     45.000, 'unidades', 5.000, 450.00)

ON CONFLICT DO NOTHING;

-- ============================================================
-- CONOCIMIENTO AGRONÓMICO (sin embeddings reales)
-- Los embeddings son vectores cero: serán reemplazados por el
-- script Python scripts/cargar_conocimiento.py
-- ============================================================

INSERT INTO conocimiento_agronomico (
    titulo_documento, tipo_cultivo_aplica, seccion,
    fragmento_texto, metadata, embedding
) VALUES

(
    'Guía de Plagas y Enfermedades del Maíz — CIMMYT 2022',
    'maiz',
    'Capítulo 3: Enfermedades Foliares',
    'La ROYA COMÚN DEL MAÍZ (Puccinia sorghi) se caracteriza por pústulas de color café-rojizo en ambas caras de la hoja. Las pústulas son ovaladas, de 1-3 mm, y producen esporas de color óxido que se dispersan por el viento. Las hojas afectadas muestran manchas cloróticas (amarillentas) antes de que aparezcan las pústulas. En ataques severos, las hojas se secan prematuramente. El hongo requiere humedad relativa mayor al 70% y temperaturas de 16-23°C para desarrollarse. TRATAMIENTO: Aplicar fungicida tebuconazol 25% a razón de 0.5 L/ha o mancozeb 80% a 2.5 kg/ha cuando se detecten las primeras pústulas. Evitar exceso de nitrógeno que favorece el crecimiento vegetativo suculento. Rotar cultivos con leguminosas.',
    '{"fuente": "CIMMYT", "año": 2022, "pagina": 47, "idioma": "es", "tags": ["roya", "hongo", "maiz", "foliar"]}',
    array_fill(0, ARRAY[1536])::vector
),

(
    'Guía de Plagas y Enfermedades del Maíz — CIMMYT 2022',
    'maiz',
    'Capítulo 4: Plagas Insectiles',
    'El GUSANO COGOLLERO (Spodoptera frugiperda) es la plaga más devastadora del maíz en América Latina. Las larvas jóvenes (L1-L2) raspan el tejido foliar dejando ventanas translúcidas ("papel encerado"). Las larvas maduras (L4-L6) perforan el cogollo y producen excrementos oscuros dentro de él — signo diagnóstico característico. En ataques tempranos (antes de V6), puede causar pérdidas del 20-73% del rendimiento. MANEJO INTEGRADO: 1) Monitoreo semanal desde emergencia. 2) Aplicar Bacillus thuringiensis (Bt) en instar 1-2 a 1.5 kg/ha. 3) Para infestaciones severas (>20% plantas con daño en cogollo): clorpirifos 480 EC a 1 L/ha + pegador. Aplicar en horas de baja temperatura (6-9 am o 5-7 pm) cuando la larva está activa en el cogollo.',
    '{"fuente": "CIMMYT", "año": 2022, "pagina": 63, "idioma": "es", "tags": ["cogollero", "insecto", "larva", "maiz"]}',
    array_fill(0, ARRAY[1536])::vector
),

(
    'Manual de Enfermedades del Frijol — INIFAP México 2021',
    'frijol',
    'Sección 2: Enfermedades Fungosas',
    'La ANTRACNOSIS DEL FRIJOL (Colletotrichum lindemuthianum) produce lesiones oscuras en vainas, tallos y hojas. En vainas: manchas hundidas de color café oscuro a negro con bordes rojizos. En hojas: lesiones en nervaduras. La enfermedad se transmite por semilla infectada y por salpicadura de lluvia. Se desarrolla en condiciones de alta humedad y temperaturas de 13-26°C. PREVENCIÓN: Usar semilla certificada libre de la enfermedad. TRATAMIENTO: Aplicar carbendazim 50% SC a razón de 0.5 L/ha o mancozeb 80% WP a 2 kg/ha. Iniciar tratamientos preventivos si las condiciones climáticas son favorables para la enfermedad (lluvia frecuente, temperaturas frescas). En caso de epidemia severa, aplicar cada 7-10 días.',
    '{"fuente": "INIFAP", "año": 2021, "pagina": 28, "idioma": "es", "tags": ["antracnosis", "hongo", "vaina", "frijol"]}',
    array_fill(0, ARRAY[1536])::vector
),

(
    'Guía de Manejo del Cultivo de Café — SAGARPA 2020',
    'cafe',
    'Capítulo 6: Roya del Café',
    'La ROYA DEL CAFÉ (Hemileia vastatrix) es la enfermedad más destructiva del cafeto. Se manifiesta como manchas amarillo-anaranjadas en el envés de las hojas, con una masa polvorienta de esporas de color naranja-amarillo. Las hojas infectadas caen prematuramente (defoliación), debilitando severamente la planta y reduciendo la producción del ciclo siguiente. El hongo se desarrolla entre 16-28°C con humedad relativa mayor al 80%. En epidemias severas puede destruir el 50-80% de la cosecha. MANEJO: Aplicar fungicida cúprico (hidróxido de cobre o oxicloruro de cobre) de forma preventiva al inicio de lluvias, cada 30-45 días. Para control curativo, usar tebuconazol 25% a 0.5 L/ha o trifloxistrobina + tebuconazol. Podar para mejorar ventilación. Fertilizar adecuadamente con K y Ca para aumentar tolerancia.',
    '{"fuente": "SAGARPA", "año": 2020, "pagina": 112, "idioma": "es", "tags": ["roya", "hongo", "cafe", "defoliacion"]}',
    array_fill(0, ARRAY[1536])::vector
),

(
    'Fertilización en Cultivos Básicos — Colegio de Postgraduados 2023',
    NULL,
    'Capítulo 1: Nutrición Mineral de los Cultivos',
    'Los síntomas de DEFICIENCIA DE NITRÓGENO en maíz se manifiestan como amarillamiento (clorosis) que inicia en la punta de las hojas más viejas (inferiores) y avanza hacia la base en forma de "V" invertida. Las plantas crecen menos, son más claras y producen menos biomasa. En frijol, las hojas amarillean uniformemente y pueden presentar necrosis. El nitrógeno es el macronutriente más importante para el crecimiento vegetativo. CORRECCIÓN: Aplicar urea 46% al suelo (50-80 kg/ha) o hacer aplicación foliar con urea al 2% para corrección rápida (respuesta en 5-7 días). Fraccionar la aplicación: 50% en siembra y 50% en V6 para maíz. Tomar muestras de suelo antes de fertilizar para ajustar dosis y evitar exceso.',
    '{"fuente": "Colegio de Postgraduados", "año": 2023, "pagina": 15, "idioma": "es", "tags": ["nitrogeno", "deficiencia", "fertilizacion", "maiz", "frijol"]}',
    array_fill(0, ARRAY[1536])::vector
);

-- Confirmación
DO $$
DECLARE
    n_socios INT;
    n_insumos INT;
    n_conocimiento INT;
BEGIN
    SELECT COUNT(*) INTO n_socios FROM socios;
    SELECT COUNT(*) INTO n_insumos FROM inventario_insumos;
    SELECT COUNT(*) INTO n_conocimiento FROM conocimiento_agronomico;
    RAISE NOTICE 'Seed 001 completado: % socios, % insumos, % fragmentos de conocimiento (embeddings pendientes).',
        n_socios, n_insumos, n_conocimiento;
END $$;
