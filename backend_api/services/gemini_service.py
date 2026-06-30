"""
backend_api/services/gemini_service.py
========================================
Servicio de integración con la API de Google Gemini.

Responsabilidades:
  1. CLASIFICAR la intención del agricultor (agronomica / entrega / insumo / mixta)
  2. EXTRAER entidades estructuradas del texto dictado
  3. GENERAR diagnóstico agronómico usando contexto RAG de pgvector
  4. Garantizar salida JSON estricta (sin alucinaciones de formato)

Anti-alucinación:
  - Se usan prompts con few-shot examples para cada intención
  - Se forza JSON válido con response_mime_type="application/json"
  - Temperatura=0.1 para respuestas deterministas
  - El campo `respuesta_para_tts` siempre en segunda persona y lenguaje campesino

Límites del nivel gratuito de Gemini Flash:
  - 15 RPM (requests por minuto)
  - 1M tokens por día
  - Con utterances de 3-10s, el pipeline puede manejar ~200 consultas/día fácilmente
"""

import json
import logging
import time
from typing import Optional, Union

import google.generativeai as genai
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shared.schemas import (
    DiagnosticoAgronomico,
    IntentionMixta,
    RegistroEntrega,
    SolicitudInsumo,
    TipoIntencion,
)

logger = logging.getLogger(__name__)

# Tipos de retorno posibles del servicio
RespuestaGemini = Union[
    DiagnosticoAgronomico,
    RegistroEntrega,
    SolicitudInsumo,
    IntentionMixta,
]

# ===========================================================
# PROMPTS DEL SISTEMA
# ===========================================================

SYSTEM_PROMPT_CLASIFICADOR = """
Eres el asistente de voz de una cooperativa agrícola en México.
Recibes transcripciones de voz de agricultores y debes clasificar su intención.

INTENCIONES POSIBLES:
- consulta_agronomica: El agricultor pregunta sobre plagas, enfermedades, cultivos o cómo tratar sus plantas.
- registrar_entrega: El agricultor quiere registrar que entregó cosecha (menciona kilos, tipo de cultivo).
- solicitar_insumo: El agricultor pide fertilizante, herbicida, semillas u otro insumo del almacén.
- mixta: El agricultor hace más de una cosa en el mismo mensaje.

Responde ÚNICAMENTE con un JSON válido. Sin texto extra, sin markdown, sin explicaciones.

EJEMPLOS:

Input: "Las hojas de mi maíz se están poniendo amarillas con manchas cafés, ¿qué le pasa?"
Output: {"intencion": "consulta_agronomica"}

Input: "Soy el socio 402, traigo 800 kilos de frijol negro"
Output: {"intencion": "registrar_entrega"}

Input: "Necesito 10 kilos de urea para mi parcela"
Output: {"intencion": "solicitar_insumo"}

Input: "Soy el socio 100, vengo a dejar 500 kilos de café y también necesito 5 litros de fungicida"
Output: {"intencion": "mixta"}
"""

SYSTEM_PROMPT_AGRONOMICO = """
Eres un ingeniero agrónomo experto que asiste a pequeños agricultores mexicanos por voz.
Tu respuesta debe ser práctica, empática y en español sencillo (no técnico).

CONTEXTO RAG (fragmentos de manuales agronómicos relevantes):
{contexto_rag}

CONSULTA DEL AGRICULTOR:
{consulta}

Responde ÚNICAMENTE con este JSON válido (sin markdown, sin texto extra):
{{
  "intencion": "consulta_agronomica",
  "plaga_o_enfermedad_probable": "nombre científico y común",
  "nivel_confianza": 0.0,
  "tratamiento_recomendado": "descripción del tratamiento con dosis concretas",
  "advertencias": ["advertencia 1", "advertencia 2"],
  "respuesta_para_tts": "texto completo en segunda persona, lenguaje campesino, máximo 100 palabras",
  "fuentes_rag": ["fuente 1", "fuente 2"]
}}

REGLAS:
- nivel_confianza entre 0.0 y 1.0
- Si el contexto RAG no es suficiente, nivel_confianza < 0.5 y recomienda consultar a un técnico
- respuesta_para_tts: natural, empático, como si hablaras con el agricultor en persona
- Menciona dosis concretas si el contexto las tiene
"""

SYSTEM_PROMPT_ENTREGA = """
Extrae los datos de entrega de cosecha de la transcripción del agricultor.

TRANSCRIPCIÓN: {texto}

Responde ÚNICAMENTE con este JSON válido:
{{
  "intencion": "registrar_entrega",
  "id_membresco_socio": "número o código del socio",
  "tipo_cultivo": "maiz|frijol|cafe|trigo|sorgo|arroz|otro",
  "cantidad_kg": 0.0,
  "calidad_estimada": "Premium|Estándar|Rechazado|null",
  "confirmacion_para_tts": "texto corto de confirmación para leer al agricultor"
}}

REGLAS:
- tipo_cultivo siempre en minúsculas y sin acentos
- Si el agricultor no menciona calidad, usar null
- confirmacion_para_tts: breve, ej: "Registré tu entrega de 800 kilos de frijol, socio 402"
- Si algún dato no está claro, usar "desconocido" y bajará cantidad_kg a 0
"""

SYSTEM_PROMPT_INSUMO = """
Extrae los datos de solicitud de insumo de la transcripción del agricultor.

TRANSCRIPCIÓN: {texto}

Responde ÚNICAMENTE con este JSON válido:
{{
  "intencion": "solicitar_insumo",
  "id_membresco_socio": "número o código del socio",
  "nombre_insumo": "nombre del insumo como lo dijo el agricultor",
  "cantidad": 0.0,
  "unidad_medida": "kg|litros|unidades|sacos",
  "confirmacion_para_tts": "texto corto de confirmación para leer al agricultor"
}}

REGLAS:
- nombre_insumo: exactamente como lo dijo el agricultor (para búsqueda fuzzy)
- unidad_medida: inferir del contexto (fertilizantes=kg, herbicidas/fungicidas=litros)
- confirmacion_para_tts: ej: "Tu solicitud de 5 kilos de urea está registrada como pendiente"
"""

SYSTEM_PROMPT_MIXTO = """
El agricultor hizo múltiples acciones en un solo mensaje.
Extrae CADA acción por separado.

TRANSCRIPCIÓN: {texto}

Responde ÚNICAMENTE con este JSON válido:
{{
  "intencion": "mixta",
  "sub_intentos": [
    {{
      "intencion": "registrar_entrega",
      "id_membresco_socio": "...",
      "tipo_cultivo": "...",
      "cantidad_kg": 0.0,
      "calidad_estimada": null,
      "confirmacion_para_tts": "..."
    }},
    {{
      "intencion": "solicitar_insumo",
      "id_membresco_socio": "...",
      "nombre_insumo": "...",
      "cantidad": 0.0,
      "unidad_medida": "...",
      "confirmacion_para_tts": "..."
    }}
  ],
  "confirmacion_para_tts": "resumen breve de todo lo que se registró"
}}
"""


# ===========================================================
# SERVICIO GEMINI
# ===========================================================

class GeminiService:
    """
    Cliente del API de Google Gemini para procesamiento de lenguaje natural.

    Gestiona la clasificación de intenciones y la generación de respuestas
    estructuradas con salida JSON estricta.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY no configurada. "
                "Obtener una gratis en https://aistudio.google.com/"
            )
        genai.configure(api_key=api_key)

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",  # Fuerza JSON válido en la respuesta
            max_output_tokens=1024,
        )

        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
        self._model_name  = model_name
        self._total_calls = 0
        self._errores     = 0
        self._latencia_acumulada_ms = 0.0

        logger.info("GeminiService inicializado: modelo='%s'", model_name)

    # ----------------------------------------------------------
    # API pública
    # ----------------------------------------------------------

    async def procesar_transcripcion(
        self,
        texto:          str,
        contexto_rag:   Optional[list[dict]] = None,
    ) -> Optional[RespuestaGemini]:
        """
        Pipeline completo de procesamiento:
        1. Clasificar intención del texto
        2. Según la intención, extraer entidades con el prompt apropiado

        Args:
            texto:        Texto transcrito del agricultor.
            contexto_rag: Fragmentos de conocimiento agronómico (solo para consultas).

        Returns:
            Modelo Pydantic correspondiente a la intención, o None en caso de error.
        """
        # Paso 1: Clasificar intención
        intencion = await self._clasificar_intencion(texto)
        if not intencion:
            logger.error("No se pudo clasificar la intención de: '%s'", texto[:80])
            return None

        logger.info("Intención detectada: %s para texto='%s…'", intencion, texto[:60])

        # Paso 2: Extraer entidades según la intención
        if intencion == TipoIntencion.AGRONOMICA:
            return await self._procesar_consulta_agronomica(texto, contexto_rag or [])
        elif intencion == TipoIntencion.ENTREGA:
            return await self._procesar_entrega(texto)
        elif intencion == TipoIntencion.INSUMO:
            return await self._procesar_insumo(texto)
        elif intencion == TipoIntencion.MIXTA:
            return await self._procesar_mixto(texto)
        else:
            logger.warning("Intención desconocida: %s", intencion)
            return None

    # ----------------------------------------------------------
    # Clasificación de intención
    # ----------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.WARNING),
    )
    async def _clasificar_intencion(self, texto: str) -> Optional[str]:
        """Clasifica la intención con Gemini. Reintenta hasta 3 veces con backoff."""
        t0 = time.perf_counter()
        try:
            prompt = f"{SYSTEM_PROMPT_CLASIFICADOR}\n\nInput: {texto}\nOutput:"
            respuesta = await self._model.generate_content_async(prompt)
            datos = self._parsear_json(respuesta.text)
            return datos.get("intencion")
        except Exception as e:
            self._errores += 1
            logger.warning("Error clasificando intención: %s", e)
            raise
        finally:
            self._registrar_latencia(t0)

    # ----------------------------------------------------------
    # Extractores por intención
    # ----------------------------------------------------------

    async def _procesar_consulta_agronomica(
        self, texto: str, contexto_rag: list[dict]
    ) -> Optional[DiagnosticoAgronomico]:
        """Genera diagnóstico agronómico usando RAG como contexto."""
        # Formatear fragmentos RAG para el prompt
        contexto_str = self._formatear_contexto_rag(contexto_rag)

        prompt = SYSTEM_PROMPT_AGRONOMICO.format(
            contexto_rag=contexto_str,
            consulta=texto,
        )

        t0 = time.perf_counter()
        try:
            respuesta = await self._modelo_call_retry(prompt)
            datos = self._parsear_json(respuesta)

            # Agregar fuentes RAG al diagnóstico
            if not datos.get("fuentes_rag") and contexto_rag:
                datos["fuentes_rag"] = [
                    f"{f.get('titulo_documento', 'Fuente desconocida')} "
                    f"(similitud: {f.get('similitud', 0):.2f})"
                    for f in contexto_rag
                ]

            return DiagnosticoAgronomico.model_validate(datos)
        except Exception as e:
            logger.error("Error en consulta agronómica: %s", e)
            return None
        finally:
            self._registrar_latencia(t0)

    async def _procesar_entrega(self, texto: str) -> Optional[RegistroEntrega]:
        """Extrae datos de entrega de cosecha."""
        prompt = SYSTEM_PROMPT_ENTREGA.format(texto=texto)
        t0 = time.perf_counter()
        try:
            respuesta = await self._modelo_call_retry(prompt)
            datos = self._parsear_json(respuesta)
            return RegistroEntrega.model_validate(datos)
        except Exception as e:
            logger.error("Error extrayendo datos de entrega: %s", e)
            return None
        finally:
            self._registrar_latencia(t0)

    async def _procesar_insumo(self, texto: str) -> Optional[SolicitudInsumo]:
        """Extrae datos de solicitud de insumo."""
        prompt = SYSTEM_PROMPT_INSUMO.format(texto=texto)
        t0 = time.perf_counter()
        try:
            respuesta = await self._modelo_call_retry(prompt)
            datos = self._parsear_json(respuesta)
            return SolicitudInsumo.model_validate(datos)
        except Exception as e:
            logger.error("Error extrayendo solicitud de insumo: %s", e)
            return None
        finally:
            self._registrar_latencia(t0)

    async def _procesar_mixto(self, texto: str) -> Optional[IntentionMixta]:
        """Extrae múltiples intenciones de un utterance mixto."""
        prompt = SYSTEM_PROMPT_MIXTO.format(texto=texto)
        t0 = time.perf_counter()
        try:
            respuesta = await self._modelo_call_retry(prompt)
            datos = self._parsear_json(respuesta)
            return IntentionMixta.model_validate(datos)
        except Exception as e:
            logger.error("Error procesando intención mixta: %s", e)
            return None
        finally:
            self._registrar_latencia(t0)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _modelo_call_retry(self, prompt: str) -> str:
        """Llama a Gemini con reintentos automáticos ante errores de cuota o red."""
        self._total_calls += 1
        respuesta = await self._model.generate_content_async(prompt)
        return respuesta.text

    def _parsear_json(self, texto: str) -> dict:
        """
        Parsea JSON de la respuesta de Gemini.
        Limpia posibles bloques ```json ... ``` si el modelo los incluye.
        """
        texto = texto.strip()
        # Remover bloques markdown si el modelo los incluye pese a response_mime_type
        if texto.startswith("```"):
            lineas = texto.split("\n")
            texto = "\n".join(
                l for l in lineas
                if not l.strip().startswith("```")
            )
        return json.loads(texto)

    def _formatear_contexto_rag(self, fragmentos: list[dict]) -> str:
        """Formatea fragmentos RAG como texto estructurado para el prompt."""
        if not fragmentos:
            return "No se encontraron fragmentos relevantes en la base de conocimiento."

        partes = []
        for i, f in enumerate(fragmentos, 1):
            partes.append(
                f"[Fragmento {i}] Fuente: {f.get('titulo_documento', 'Desconocida')} "
                f"(similitud: {f.get('similitud', 0):.2f})\n"
                f"{f.get('fragmento_texto', '')}"
            )
        return "\n\n---\n\n".join(partes)

    def _registrar_latencia(self, t0: float) -> None:
        self._latencia_acumulada_ms += (time.perf_counter() - t0) * 1000

    @property
    def latencia_promedio_ms(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._latencia_acumulada_ms / self._total_calls

    def log_metricas(self) -> None:
        logger.info(
            "📊 Gemini — llamadas: %d | errores: %d | latencia_prom: %.0fms | modelo: %s",
            self._total_calls, self._errores,
            self.latencia_promedio_ms, self._model_name,
        )
