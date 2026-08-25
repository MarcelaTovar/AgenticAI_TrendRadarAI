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
