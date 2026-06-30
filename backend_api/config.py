"""
backend_api/config.py
=======================
Configuración tipada del backend API usando Pydantic Settings.

Pydantic Settings lee automáticamente desde variables de entorno y/o
archivos .env. Todos los campos tienen defaults para desarrollo local.

Importar en cualquier módulo del backend:
    from config import settings
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración global del backend AgroVoz.
    Los nombres de campo coinciden exactamente con las variables del .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Base de datos ---
    database_url: str = Field(
        default="postgresql+asyncpg://agrovoz:agrovoz_secret_change_me@localhost:5432/agrovoz_db",
        description="URL async de PostgreSQL con asyncpg.",
    )
    db_pool_size:     int = Field(default=10, ge=1, le=50)
    db_max_overflow:  int = Field(default=5,  ge=0, le=20)
    db_pool_timeout:  int = Field(default=30, ge=5)

    # --- Redis ---
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="URL completa de Redis con contraseña.",
    )
    redis_stream_texto:     str = Field(default="agrovoz:texto")
    redis_stream_respuesta: str = Field(default="agrovoz:respuesta")
    redis_group_backend:    str = Field(default="grupo_backend")

    # --- Gemini API ---
    gemini_api_key: str = Field(
        default="",
        description="API Key de Google AI Studio (nivel gratuito).",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Modelo Gemini. gemini-1.5-flash = más rápido y gratuito.",
    )
    # Temperatura baja para respuestas deterministas y sin alucinaciones
    gemini_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    gemini_max_tokens:  int   = Field(default=1024, ge=100, le=8192)

    # --- RAG (pgvector) ---
    rag_top_k:      int   = Field(default=3, ge=1, le=10, description="Fragmentos a recuperar.")
    embedding_dim:  int   = Field(default=1536, description="Dimensión del modelo de embedding.")
    # Umbral mínimo de similitud coseno para incluir un fragmento en el contexto RAG
    rag_min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Embeddings ---
    # Modelo local de sentence-transformers para generar embeddings en español
    # Debe generar vectores de 'embedding_dim' dimensiones
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Modelo HuggingFace para embeddings. Soporta español.",
    )

    # --- Servidor ---
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    environment:  str = Field(default="development")
    log_level:    str = Field(default="INFO")

    @field_validator("gemini_api_key")
    @classmethod
    def validar_gemini_key(cls, v: str) -> str:
        # En producción, la key debe estar configurada
        # En tests unitarios, puede estar vacía
        return v.strip()

    @property
    def es_produccion(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def es_desarrollo(self) -> bool:
        return self.environment.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna la instancia singleton de Settings.
    El caché LRU garantiza que solo se lee el .env una vez por proceso.
    """
    return Settings()


# Alias conveniente para importaciones directas
settings = get_settings()
