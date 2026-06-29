# 🌾 AgroVoz — Asistente de Voz para Cooperativas Agrícolas

Pipeline de voz **asíncrono, resiliente y de costo cero** que permite a agricultores
dictar consultas agronómicas y trámites de cooperativa mediante voz en español,
optimizado para zonas con conectividad intermitente.

---

## Arquitectura del Sistema

```
[Micrófono]
    │ PCM 16kHz
    ▼
[cliente_ingesta]  ◄── Silero VAD (detección de voz local)
    │ audio_chunk (bytes b64) → Redis Stream "agrovoz:audio"
    ▼
[servicio_stt]     ◄── Whisper via CTranslate2 (local, sin costo)
    │ texto transcrito → Redis Stream "agrovoz:texto"
    ▼
[backend_api]      ◄── FastAPI + Gemini (gratis) + pgvector RAG
    │ JSON estructurado (Pydantic)
    ▼
[PostgreSQL + pgvector]   (datos relacionales + vectores agronómicos)
    │
    ▼
[servicio_tts]     ◄── Fábrica: Piper TTS (local) | ElevenLabs (fallback)
    │ audio WAV
    ▼
[Altavoz]
```

---

## Estructura del Repositorio

```
agrovoz/
├── README.md                    ← Este archivo
├── .env.example                 ← Variables de entorno (copiar a .env)
├── podman-compose.yml           ← Orquestación de contenedores rootless
│
├── infra/                       ← Infraestructura: DB, Redis, configs
│   └── db/
│       ├── migrations/          ← Scripts SQL ordenados (001_, 002_, …)
│       └── seeds/               ← Datos iniciales de inventario y conocimiento
│
├── cliente_ingesta/             ← PRIMER TERCIO: Captura de audio + VAD + Redis
│   ├── Containerfile
│   ├── requirements.txt
│   ├── main.py                  ← Entry point del cliente
│   ├── audio_capture.py         ← Captura PCM 16kHz con PyAudio
│   ├── vad_processor.py         ← Silero VAD: detecta voz y corta en silencios
│   └── redis_publisher.py       ← Publica chunks a Redis Stream con resiliencia
│
├── servicio_stt/                ← SEGUNDO TERCIO: Whisper + consumer Redis
│   └── (desarrollo en siguiente fase)
│
├── backend_api/                 ← SEGUNDO TERCIO: FastAPI + Gemini + RAG
│   └── (desarrollo en siguiente fase)
│
├── servicio_tts/                ← TERCER TERCIO: Fábrica Piper/ElevenLabs
│   └── (desarrollo en siguiente fase)
│
└── shared/                      ← Código compartido entre servicios
    ├── schemas.py               ← Modelos Pydantic (fuente única de verdad)
    └── redis_config.py          ← Nombres de streams y grupos de consumidores
```

---

## Inicio Rápido

```bash
# 1. Copiar variables de entorno
cp .env.example .env
# Editar .env con tu API key de Gemini y ElevenLabs

# 2. Levantar infraestructura (PostgreSQL + Redis)
podman-compose up -d postgres redis

# 3. Ejecutar migraciones
podman-compose run --rm postgres psql -U agrovoz -d agrovoz_db -f /migrations/001_schema.sql

# 4. Levantar pipeline completo
podman-compose up
```

---

## Tercios de Desarrollo

| Tercio | Componentes | Estado |
|--------|------------|--------|
| **1° Tercio** | Infraestructura + DB + Cliente Ingesta (VAD→Redis) | ✅ Completo |
| **2° Tercio** | STT (Whisper) + Backend API (FastAPI+Gemini+RAG) | 🔄 Pendiente |
| **3° Tercio** | TTS (Piper+ElevenLabs) + Tests de Estrés (Faker) | 🔄 Pendiente |

---

## Directrices Clave

- **Cero costo operativo:** Piper + Whisper son locales. ElevenLabs y Gemini usan nivel gratuito.
- **Resiliencia ante red intermitente:** el cliente reintenta conexión a Redis automáticamente.
- **Contenedores rootless:** todo corre con Podman sin privilegios de root.
- **Objetivo de latencia:** búsqueda pgvector < 50ms bajo 1M de vectores.
