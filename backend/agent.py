import os
import re
import json
from datetime import datetime, timezone
from typing import List, TypedDict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pytrends.request import TrendReq
from ddgs import DDGS
import time
import operator
from typing import Annotated

load_dotenv()

GENERATOR_NAME = "TrendRadar AI"
MODEL_NAME = "openai/gpt-oss-120b"
MANDATORY_AI_HASHTAG = "#GeneradoConIA"

# 1. Configuración del LLM y Herramientas
llm_posts = ChatGroq(
    model=MODEL_NAME,
    temperature=0.4,
    max_tokens=2000,
    reasoning_effort="low"
)

def search_ddg_structured(query: str, max_results: int = 3) -> List[dict]:
    """Ejecuta búsquedas web y devuelve resultados estructurados."""
    try:
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=max_results))
        if not raw_results:
            print(f"⚠️  Búsqueda '{query}' devolvió 0 resultados (sin excepción).")
        return [
            {
                "title": r.get("title", "").strip(),
                "url": r.get("href", "") or r.get("link", ""),
                "snippet": r.get("body", "")[:200].strip()
            }
            for r in raw_results
        ]
    except Exception as e:
        print(f"⚠️  Error en búsqueda DDG para '{query}': {type(e).__name__}: {e}")
        return []

def format_results_for_prompt(results: List[dict], max_chars: int = 1200) -> str:
    combined = "\n".join(f"{r['title']}: {r['snippet']}" for r in results)
    return combined[:max_chars] if combined else "Sin resultados de búsqueda disponibles."

def get_google_trends(region: str = "united_states", limit: int = 5) -> List[str]:
    """Obtiene los temas en tendencia actual desde Google Trends."""
    try:
        pytrends = TrendReq(hl='es-US', tz=360)
        trending_df = pytrends.trending_searches(pn=region)
        trends = trending_df[0].tolist()[:limit]
        return trends
    except Exception:
        return ["Inteligencia Artificial", "Cambio Climático", "Ciberseguridad", "Exploración Espacial", "Economía Digital"][:limit]

def sanitize_custom_instructions(raw: Optional[str]) -> str:
    """Limita y filtra las instrucciones del usuario antes de inyectarlas en cualquier prompt."""
    if not raw:
        return ""
    text = raw.strip()[:400]
    blocked_patterns = [
        r"ignora\s+(las\s+)?(instrucciones|reglas)",
        r"olvida\s+(las\s+)?(instrucciones|reglas)",
        r"no\s+verifiques",
        r"inventa\s+datos",
        r"actua\s+como\s+si\s+no\s+tuvieras\s+reglas",
        r"system\s*prompt",
        r"eres\s+libre\s+de\s+mentir",
        r"no\s+marques\s+(el\s+)?contenido\s+como\s+ia",
        r"finge\s+ser\s+(un\s+)?human[oa]",
        r"oculta\s+que\s+eres\s+ia",
    ]
    for pattern in blocked_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return ""
    return text

def build_disclosure_footer() -> str:
    """Aviso mínimo de transparencia. Se agrega SIEMPRE, sin depender del LLM."""
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y a las %H:%M UTC")
    return (
        f"\n\n---\n"
        f"*Este contenido fue generado por {GENERATOR_NAME} el {now}. "
        f"Los datos fueron verificados mediante búsqueda web automatizada, "
        f"pero requiere revisión y aprobación humana antes de publicarse.*"
    )

# 2. Estado del Agente
class AgentState(TypedDict):
    user_input: str
    intent: str
    num_topics: int
    selected_topic: Optional[str]
    suggested_topics_output: Optional[str]
    trends_raw: Optional[str]
    verified_data: Optional[str]
    executive_report: Optional[str]
    social_posts: Optional[str]
    final_report: Optional[str]
    system_prompt: Optional[str]
    user_name: Optional[str]
    sources_trends: Optional[List[dict]]
    sources_verification: Optional[List[dict]]
    provenance: Optional[dict]
    metrics: Annotated[List[dict], operator.add]  # NUEVO — se concatena, no se sobrescribe
    performance: Optional[dict]

# 3. Nodos del Grafo

def classify_intent(state: AgentState) -> dict:
    print("[Router] Clasificando la solicitud...")
    start = time.perf_counter()
    user_text = state['user_input'].lower()

    if any(k in user_text for k in ["opcion 1", "opción 1", "sugiere", "dame", "opciones", "temas para elegir"]):
        metric = make_metric("classify_intent", start, extra={"method": "regla_directa"})
        return {"intent": "suggest_topics", "metrics": [metric]}
    elif any(k in user_text for k in ["opcion 2", "opción 2", "escoge tu", "elige tu", "escoge tú", "hazlo tu"]):
        metric = make_metric("classify_intent", start, extra={"method": "regla_directa"})
        return {"intent": "auto_select", "metrics": [metric]}

    prompt = f"""
    Clasifica la siguiente solicitud de usuario en UNA de las 3 categorías exactas:

    Solicitud: "{state['user_input']}"

    Categorías:
    - SUGGEST: El usuario quiere ver una lista de N temas virales con descripción para él escoger.
    - AUTO_SELECT: El usuario pide que el agente escoja autónomamente el mejor tema y genere el reporte completo.
    - DIRECT: El usuario especifica un tema concreto (ej: "IA", "Bitcoin").

    Responde ÚNICAMENTE con la palabra en mayúsculas: SUGGEST, AUTO_SELECT o DIRECT.
    """
    response = llm_posts.invoke(prompt)          # objeto completo, no .content todavía
    raw = response.content.strip().upper()        # el texto se extrae aquí, una sola vez

    if "SUGGEST" in raw:
        intent = "suggest_topics"
    elif "AUTO_SELECT" in raw:
        intent = "auto_select"
    else:
        intent = "direct_topic"

    metric = make_metric("classify_intent", start, response, extra={"method": "llm"})
    return {"intent": intent, "metrics": [metric]}

def handle_suggest(state: AgentState) -> dict:
    n = state.get("num_topics", 3)
    print(f"[Google Trends] Obteniendo {n} temas virales...")
    start = time.perf_counter()
    trends_list = get_google_trends(limit=n)
    prompt = f"""
    Los siguientes temas están en tendencia en Google Trends:
    {', '.join(trends_list)}

    Para cada uno de los {n} temas, presenta un formato limpio con:
    1. **Nombre del tema**
    2. Breve descripción (1-2 oraciones) de por qué es noticia o tendencia.

    Sé conciso y claro.
    """
    response = llm_posts.invoke(prompt)
    metric = make_metric("handle_suggest", start, response)
    return {"suggested_topics_output": str(response.content), "metrics": [metric]}

def handle_auto_select(state: AgentState) -> dict:
    print("[Google Trends] Seleccionando autónomamente el mejor tema...")
    start = time.perf_counter()
    trends_list = get_google_trends(limit=5)


    prompt = f"""
    De esta lista de temas virales: {', '.join(trends_list)}
    Selecciona EL TEMA MÁS RELEVANTE para hacer un reporte ejecutivo.
    Devuelve ÚNICAMENTE el nombre del tema seleccionado.
    """
    response = llm_posts.invoke(prompt)
    selected = response.content.strip()
    print(f"🤖 [Agente] Tema seleccionado: '{selected}'")

    metric = make_metric("handle_auto_select", start, response)
    return {"selected_topic": selected, "metrics": [metric]}

def set_direct_topic(state: AgentState) -> dict:
    return {"selected_topic": state["user_input"]}

def find_trends(state: AgentState) -> dict:
    topic = state.get("selected_topic", "Tendencias Tecnológicas")
    print(f"[Buscador] Investigando tendencias sobre: {topic}...")
    start = time.perf_counter()
    query = f"{topic} noticias tendencias recientes 2026"
    results = search_ddg_structured(query)
    search_results = format_results_for_prompt(results)

    prompt = f"""
    A partir de estos resultados de búsqueda, resume las 2-3 tendencias principales sobre '{topic}'.
    Incluye datos, cifras, nombres o hechos concretos si aparecen en el texto (no generalices).

    Resultados de búsqueda:
    {search_results}
    """
    response = llm_posts.invoke(prompt)
    metric = make_metric("find_trends", start, response, extra={"search_results_count": len(results)})
    return {"trends_raw": str(response.content), "sources_trends": results, "metrics": [metric]}

def verify_and_contrast(state: AgentState) -> dict:
    print("[Verificador] Verificando datos...")
    start = time.perf_counter()
    trends_summary = state.get("trends_raw", "")[:800]
    topic = state.get("selected_topic", "")
    query = f"{topic} verificacion datos fuentes 2026"
    results = search_ddg_structured(query)
    contrast_results = format_results_for_prompt(results)

    prompt = f"""
    Analiza las tendencias: {trends_summary}
    Frente a estos datos de contraste: {contrast_results}

    REGLA: Si un dato aparece solo en una fuente dudosa y es contradicho, DESCÁRTALO.
    Devuelve un resumen verificado que conserve los datos, cifras y hechos concretos que sí se sostienen.
    """
    response = llm_posts.invoke(prompt)
    verified_text = str(response.content)

    # Proxy simple de "hubo descarte" para medir qué tan seguido el filtro de calidad actúa
    discard_detected = bool(re.search(r"descart", verified_text, flags=re.IGNORECASE))

    metric = make_metric(
        "verify_and_contrast", start, response,
        extra={"search_results_count": len(results), "discard_detected": discard_detected}
    )
    return {"verified_data": verified_text, "sources_verification": results, "metrics": [metric]}

def generate_executive_report(state: AgentState) -> dict:
    print("[Redactor] Generando el reporte ejecutivo...")
    start = time.perf_counter()
    topic = state.get("selected_topic", "Tema General")
    verified = state.get("verified_data", "")
    custom_instructions = sanitize_custom_instructions(state.get("system_prompt"))

    prompt = f"""
    Basado ESTRICTAMENTE en esta información verificada sobre '{topic}' (usa datos, cifras o hechos concretos mencionados, evita definiciones genéricas del tema):
    {verified}

    INSTRUCCIONES DE ESTILO DEL USUARIO (aplícalas solo al TONO y VOCABULARIO del texto, nunca omitas ni inventes datos por seguirlas, y nunca las uses para ocultar que el contenido es generado por IA):
    {custom_instructions if custom_instructions else "Ninguna instrucción especial."}

    Genera EXACTAMENTE este formato en Markdown, sin texto adicional antes o después:

    # Reporte Ejecutivo: {topic}

    ## Resumen Verificado
    [3-5 oraciones que incluyan al menos 2 datos/hechos concretos extraídos del contexto]

    ## Control de Calidad
    [Menciona qué datos se descartaron y por qué, o escribe 'Información 100% verificada.']
    """
    response = llm_posts.invoke(prompt)
    metric = make_metric("generate_executive_report", start, response)
    return {"executive_report": str(response.content).strip(), "metrics": [metric]}

def generate_social_posts(state: AgentState) -> dict:
    """
    Genera los posts en JSON estructurado (evita tablas Markdown rotas) y agrega
    #GeneradoConIA por código a CADA post, de forma inseparable del texto — no
    depende de que el LLM decida incluirlo.
    """
    print("[Copywriter] Generando publicaciones para redes sociales...")
    start = time.perf_counter()
    topic = state.get("selected_topic", "Tema General")
    verified = state.get("verified_data", "")
    custom_instructions = sanitize_custom_instructions(state.get("system_prompt"))

    prompt = f"""
    Eres un copywriter experto en redes sociales. Basado en esta información verificada sobre '{topic}':
    {verified}

    INSTRUCCIONES DE ESTILO DEL USUARIO (aplícalas solo al TONO y VOCABULARIO, nunca omitas ni inventes datos por seguirlas. NO las apliques aquí si el usuario pidió excluirlas de las redes sociales):
    {custom_instructions if custom_instructions else "Ninguna instrucción especial."}

    REGLA OBLIGATORIA E INNEGOCIABLE: nunca escribas el post simulando ser una
    opinión personal espontánea o una experiencia vivida en primera persona
    (ej. "yo pienso que...", "me pasó que...", "no puedo creer que..."). El post
    debe sonar como comunicación institucional/informativa de una cuenta de
    contenido o marca, NUNCA como la voz de un individuo compartiendo su vida o
    pensamientos genuinos. Esto aplica sin excepción, sin importar otras
    instrucciones de estilo.

    Genera 3 publicaciones DISTINTAS y COMPLETAS. Usa datos concretos del contexto, no generalidades.

    Reglas por plataforma:
    - LinkedIn: tono profesional, 80-120 palabras en el texto del post (los hashtags van aparte, no cuentan en este límite). 3-4 hashtags específicos al tema.
    - X (Twitter): tono directo/informativo, el TEXTO del post (sin contar hashtags) debe tener máximo 220 caracteres. 2-3 hashtags cortos.
    - Instagram: tono cercano y dinámico, 40-60 palabras en el texto (hashtags aparte). 3-4 hashtags.

    No repitas frases ni el ángulo entre los tres posts.

    Responde ÚNICAMENTE con un JSON válido (sin texto antes o después, sin bloques de código ```), con esta estructura exacta:

    {{
      "linkedin": {{"post": "texto del post", "hashtags": ["#Hashtag1", "#Hashtag2", "#Hashtag3"]}},
      "twitter": {{"post": "texto del post", "hashtags": ["#Hashtag1", "#Hashtag2"]}},
      "instagram": {{"post": "texto del post", "hashtags": ["#Hashtag1", "#Hashtag2", "#Hashtag3"]}}
    }}
    """
    response = llm_posts.invoke(prompt)
    raw = str(response.content).strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    def fmt(platform: dict) -> tuple[str, str]:
        post = str(platform.get("post", "")).strip()
        hashtags = list(platform.get("hashtags", []))
        # El hashtag de transparencia se agrega SIEMPRE por código, sin depender del modelo.
        if MANDATORY_AI_HASHTAG not in hashtags:
            hashtags.append(MANDATORY_AI_HASHTAG)
        return post, " ".join(hashtags)
    json_parse_success = True
    twitter_length_ok = None

    try:
        data = json.loads(raw)
        li_post, li_tags = fmt(data.get("linkedin", {}))
        tw_post, tw_tags = fmt(data.get("twitter", {}))
        ig_post, ig_tags = fmt(data.get("instagram", {}))
        twitter_length_ok = len(tw_post) <= 220 

        content = (
            "## Publicaciones para Redes Sociales\n\n"
            "| Red Social | Texto del Post | Hashtags Recomendados |\n"
            "| :--- | :--- | :--- |\n"
            f"| LinkedIn | {li_post} | {li_tags} |\n"
            f"| X (Twitter) | {tw_post} | {tw_tags} |\n"
            f"| Instagram | {ig_post} | {ig_tags} |"
        )
    except (json.JSONDecodeError, AttributeError) as e:
        json_parse_success = False
        print(f"⚠️  ADVERTENCIA: no se pudo parsear el JSON de posts ({e}). Respuesta cruda:\n{raw[:300]}")
        content = "## Publicaciones para Redes Sociales\n\n_No se pudieron generar los posts en este intento. Intenta de nuevo._"

    metric = make_metric(
        "generate_social_posts", start, response,
        extra={"json_parse_success": json_parse_success, "twitter_length_ok": twitter_length_ok}
    )
    return {"social_posts": content, "metrics": [metric]}

def assemble_final_report(state: AgentState) -> dict:
    print("[Ensamblador] Uniendo reporte, publicaciones y metadata de transparencia...")
    user_name = (state.get("user_name") or "Usuario").strip()[:60]
    topic = state.get("selected_topic", "Tema General")

    saludo = f"¡Hola, {user_name}! Aquí tienes tu reporte:\n\n"
    disclosure = build_disclosure_footer()
    combined = (
        f"{saludo}"
        f"{state.get('executive_report', '')}\n\n"
        f"{state.get('social_posts', '')}"
        f"{disclosure}"
    )

    provenance = {
        "ai_generated": True,
        "generator": GENERATOR_NAME,
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "sources_trends": state.get("sources_trends", []),
        "sources_verification": state.get("sources_verification", []),
        "quality_control_summary": state.get("verified_data", "")
    }

    # Agregación de métricas de toda la ejecución
    all_metrics = state.get("metrics", [])
    total_duration = round(sum(m.get("duration_seconds", 0) for m in all_metrics), 3)
    total_tokens = sum(m.get("total_tokens") or 0 for m in all_metrics)
    llm_calls = sum(1 for m in all_metrics if m.get("total_tokens") is not None)

    social_metric = next((m for m in all_metrics if m["node"] == "generate_social_posts"), {})
    verify_metric = next((m for m in all_metrics if m["node"] == "verify_and_contrast"), {})

    performance = {
        "total_duration_seconds": total_duration,
        "total_tokens": total_tokens,
        "llm_calls": llm_calls,
        "nodes_executed": len(all_metrics),
        "json_parse_success": social_metric.get("json_parse_success"),
        "twitter_length_ok": social_metric.get("twitter_length_ok"),
        "discard_detected_in_verification": verify_metric.get("discard_detected"),
        "per_node": all_metrics
    }

    # --- Escritura de métricas de rendimiento (independiente) ---
    try:
        with open("metrics_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"topic": topic, "timestamp": datetime.now(timezone.utc).isoformat(), **performance},
                ensure_ascii=False
            ) + "\n")
    except Exception as e:
        print(f"⚠️  No se pudo escribir metrics_log.jsonl: {e}")

    # --- Archivo de contenido para evaluación offline (independiente) ---
    try:
        report_record = {
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verified_data": state.get("verified_data", ""),
            "executive_report": state.get("executive_report", ""),
            "social_posts": state.get("social_posts", ""),
        }
        with open("reports_archive.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(report_record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️  No se pudo escribir reports_archive.jsonl: {e}")

    return {"final_report": combined, "provenance": provenance, "performance": performance}
# 4. Enrutador Lógico
def route_intent(state: AgentState) -> str:
    return state["intent"]

def usage_from_response(response) -> dict:
    """Extrae el conteo de tokens de la respuesta de ChatGroq, si está disponible."""
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

def make_metric(node_name: str, start_time: float, response=None, extra: Optional[dict] = None) -> dict:
    """Construye una entrada de métrica estandarizada para un nodo del grafo."""
    entry = {
        "node": node_name,
        "duration_seconds": round(time.perf_counter() - start_time, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if response is not None:
        entry.update(usage_from_response(response))
    if extra:
        entry.update(extra)
    return entry



# 5. Construcción del Grafo
workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("handle_suggest", handle_suggest)
workflow.add_node("handle_auto_select", handle_auto_select)
workflow.add_node("set_direct_topic", set_direct_topic)
workflow.add_node("find_trends", find_trends)
workflow.add_node("verify_and_contrast", verify_and_contrast)
workflow.add_node("generate_executive_report", generate_executive_report)
workflow.add_node("generate_social_posts", generate_social_posts)
workflow.add_node("assemble_final_report", assemble_final_report)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "suggest_topics": "handle_suggest",
        "auto_select": "handle_auto_select",
        "direct_topic": "set_direct_topic"
    }
)

workflow.add_edge("handle_suggest", END)

workflow.add_edge("handle_auto_select", "find_trends")
workflow.add_edge("set_direct_topic", "find_trends")

workflow.add_edge("find_trends", "verify_and_contrast")

workflow.add_edge("verify_and_contrast", "generate_executive_report")
workflow.add_edge("generate_executive_report", "generate_social_posts")
workflow.add_edge("generate_social_posts", "assemble_final_report")
workflow.add_edge("assemble_final_report", END)

app = workflow.compile()

# 6. Ejecución Interactiva
if __name__ == "__main__":
    print("=== AGENTE INVESTIGADOR DE TENDENCIAS ===\n")
    print("Opciones disponibles:")
    print(" 1. 'Dame 3 temas virales para elegir' (o escribe 'Opción 1')")
    print(" 2. 'Escoge tú el tema más viral' (o escribe 'Opción 2')")
    print(" 3. 'Investiga sobre [Tu tema]'\n")

    solicitud = input("Escribe tu instrucción: ")

    solicitud_limpia = re.sub(r'opci[oó]n\s*\d+', '', solicitud, flags=re.IGNORECASE)
    match = re.search(r'\b(\d+)\b', solicitud_limpia)
    num_opciones = int(match.group(1)) if match else 3

    resultado = app.invoke({
        "user_input": solicitud,
        "num_topics": num_opciones,
        "system_prompt": None,
        "user_name": "Usuario"
    })

    if resultado.get("suggested_topics_output"):
        print("\n=== TEMAS SUGERIDOS PARA ELEGIR ===")
        print(resultado["suggested_topics_output"])
        print("\n👉 Copia el nombre de uno de estos temas y vuelve a ejecutar para armar el reporte.")

    elif resultado.get("final_report"):
        nombre_archivo = "reporte_tendencias.md"
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write(resultado["final_report"])

        print(f"\n✅ ¡Reporte completado! Guardado en '{nombre_archivo}'.")
        print(f"\n📋 Log de procedencia:\n{json.dumps(resultado.get('provenance', {}), indent=2, ensure_ascii=False)}")