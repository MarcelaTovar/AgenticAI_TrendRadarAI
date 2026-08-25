# TrendRadar AI

Sistema agéntico de investigación de tendencias construido con **LangGraph**, **LangChain** y modelos de lenguaje alojados en **Groq**. Automatiza el flujo completo de: identificación de un tema, investigación de tendencias mediante búsqueda web, verificación y contraste de la información recopilada, y generación de contenido en dos formatos — un reporte ejecutivo en Markdown y publicaciones adaptadas para LinkedIn, X e Instagram.

Proyecto desarrollado para la clase de Minería de Datos — UNITEC, por Marcela Tovar.

---

## Características principales

- **Clasificación de intención**: detecta si el usuario quiere sugerencias de temas, auto-selección autónoma, o investigación directa de un tema específico.
- **Investigación con verificación en dos pasos**: cada tendencia identificada pasa por una segunda ronda de búsqueda independiente antes de considerarse confiable, descartando datos que solo aparecen en fuentes dudosas y son contradichos.
- **Generación estructurada**: los posts de redes sociales se generan en JSON y se ensamblan en Python, evitando tablas Markdown corruptas.
- **Personalización segura**: instrucciones de estilo del usuario, sanitizadas contra intentos de *prompt injection* que busquen anular las reglas internas del sistema.
- **Componente ético incorporado**:
  - Disclosure automático de transparencia en cada reporte.
  - Log de procedencia (fuentes, modelo, timestamp, resumen de control de calidad).
  - Aprobación humana obligatoria antes de poder copiar o descargar el contenido.
  - Hashtag `#GeneradoConIA` inseparable del contenido, añadido por código.
- **Métricas de rendimiento**: latencia, tokens consumidos, tasa de éxito de parseo JSON y cumplimiento de formato, por nodo del grafo.
- **Evaluación offline de calidad (LLM-as-judge)**: proceso por lotes, desacoplado del camino crítico del usuario, que puntúa fidelidad, relevancia y completitud de cada reporte usando un modelo local vía Ollama.

---

## Arquitectura

- **Framework de orquestación**: LangGraph (grafo de estados con nodos y transiciones condicionales).
- **Modelo LLM (producción)**: `openai/gpt-oss-120b` vía API de Groq.
- **Modelo LLM (evaluación offline)**: `llama3.2:3b-instruct-fp16` vía Ollama, local.
- **Motor de búsqueda**: DuckDuckGo (`ddgs`).
- **Tendencias**: Google Trends (`pytrends`), con lista de respaldo si la API falla.
- **Backend**: FastAPI (Python).
- **Frontend**: React + Tailwind CSS.
- **Persistencia de chats**: IndexedDB en el cliente.
- **Persistencia de métricas y contenido**: archivos `.jsonl` locales.

---

## Estructura del flujo (grafo)
classify_intent
│
├── suggest_topics ──> handle_suggest ──> END
├── auto_select ─────> handle_auto_select ──┐
└── direct_topic ────> set_direct_topic ────┤
▼
find_trends
│
verify_and_contrast
│
generate_executive_report
│
generate_social_posts
│
assemble_final_report
│
END


---

## Requisitos previos

- Python 3.11+
- Node.js 18+
- Cuenta y API key de [Groq](https://console.groq.com)
- [Ollama](https://ollama.com) instalado (para la evaluación offline)

---

## Instalación

### Backend

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd trendradar-ai

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install langchain-groq langgraph ddgs pytrends python-dotenv fastapi uvicorn langchain-ollama

# Configurar variables de entorno
echo "GROQ_API_KEY=tu_api_key_aqui" > .env
```

### Modelo local para evaluación offline

```bash
ollama pull llama3.2:3b-instruct-fp16
```

### Frontend

```bash
cd frontend
npm install
```

---

## Uso

### Ejecutar el backend (API)

```bash
python server.py
```

El servidor corre en `http://localhost:8000`.

### Ejecutar el frontend

```bash
cd frontend
npm run dev
```

La app corre en `http://localhost:5173`.

### Ejecutar el agente por consola (sin frontend)

```bash
python agent.py
```

Opciones disponibles al ejecutar:
'Dame 3 temas virales para elegir' (o escribe 'Opción 1')
'Escoge tú el tema más viral' (o escribe 'Opción 2')
'Investiga sobre [Tu tema]'


### Consultar métricas de rendimiento agregadas

```bash
curl http://localhost:8000/api/metrics/summary
```

O desde la interfaz, mediante el ícono de gráfica de barras en el header.

### Correr la evaluación offline de calidad

Con el backend habiendo generado al menos un reporte (esto crea `reports_archive.jsonl` automáticamente):

```bash
python evaluate_batch.py
```

Esto genera `evaluation_results.jsonl` con la puntuación de fidelidad, relevancia y completitud de cada reporte evaluado, además de un resumen agregado impreso en consola.

---

## Archivos generados en tiempo de ejecución

| Archivo | Contenido |
|---|---|
| `metrics_log.jsonl` | Métricas de rendimiento por ejecución (latencia, tokens, tasas de éxito). |
| `reports_archive.jsonl` | Contenido completo de cada reporte generado, para evaluación offline. |
| `evaluation_results.jsonl` | Resultados de la evaluación de calidad por lote (LLM-as-judge). |
| `approval_log.jsonl` | Registro de aprobaciones humanas confirmadas desde el frontend. |

---

## Consideraciones éticas

Este proyecto incorpora salvaguardas diseñadas para mitigar dos riesgos identificados durante el desarrollo:

1. **Error no detectado**: el sistema de verificación no es infalible; se documentó un caso real de alucinación (datos e información fabricada con apariencia de fuente verificada) durante las pruebas, disponible como evidencia en `reports_archive.jsonl`.
2. **Engaño de origen**: riesgo de que el público confunda contenido generado por IA con la opinión genuina y espontánea de una persona.

Las medidas de mitigación (disclosure, log de procedencia, aprobación humana obligatoria y etiquetado inseparable) se diseñaron en línea con los principios de transparencia y trazabilidad del **Artículo 50 de la Ley de IA de la Unión Europea**.

Para más detalle, ver `documentacion_trendradar_ai.md`.

---

## Limitaciones conocidas

- La verificación de datos depende de la calidad de los resultados de búsqueda de DuckDuckGo, que puede fallar por rate-limiting.
- El modelo evaluador local (`llama3.2:3b`) tiene menor capacidad de juicio matizado que modelos más grandes; se recomienda `qwen2.5:7b` si el hardware lo permite y se busca mayor precisión en la detección de alucinaciones.
- El sistema no bloquea automáticamente la publicación de contenido con baja puntuación de fidelidad; la evaluación es informativa/offline, no un filtro en el camino crítico.

---

## Autora

Marcela Tovar — Ingeniería en Ciencia de Datos e Inteligencia Artificial, UNITEC.
