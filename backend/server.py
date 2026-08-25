import re
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importamos la app compilada de LangGraph desde tu agent.py
from agent import app as agent_app

app = FastAPI(title="TrendRadar AI API")

# Configuración de CORS para permitir peticiones desde React (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    system_prompt: Optional[str] = None
    user_name: Optional[str] = "Usuario"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        solicitud = request.user_input.strip()
        solicitud_limpia = re.sub(r'opci[oó]n\s*\d+', '', solicitud, flags=re.IGNORECASE)
        match = re.search(r'\b(\d+)\b', solicitud_limpia)
        num_opciones = int(match.group(1)) if match else 3

        resultado = agent_app.invoke({
            "user_input": solicitud,
            "num_topics": num_opciones,
            "system_prompt": request.system_prompt,
            "user_name": request.user_name
        })

        return {
            "intent": resultado.get("intent"),
            "selected_topic": resultado.get("selected_topic"),
            "suggested_topics_output": resultado.get("suggested_topics_output"),
            "final_report": resultado.get("final_report"),
            "provenance": resultado.get("provenance"),
            "performance": resultado.get("performance")   # NUEVO
        }

    except Exception as e:
        print(f"Error procesando solicitud: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/summary")
async def metrics_summary():
    """Resumen agregado de todas las ejecuciones registradas, para tu presentación/reporte."""
    import json as _json
    from pathlib import Path

    log_path = Path("metrics_log.jsonl")
    if not log_path.exists():
        return {"runs": 0, "message": "Aún no hay ejecuciones registradas."}

    runs = [_json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not runs:
        return {"runs": 0, "message": "Aún no hay ejecuciones registradas."}

    n = len(runs)
    avg_duration = sum(r.get("total_duration_seconds", 0) for r in runs) / n
    avg_tokens = sum(r.get("total_tokens", 0) for r in runs) / n
    json_success_rate = sum(1 for r in runs if r.get("json_parse_success")) / n
    twitter_compliance_rate = sum(1 for r in runs if r.get("twitter_length_ok")) / n
    discard_rate = sum(1 for r in runs if r.get("discard_detected_in_verification")) / n

    return {
        "runs": n,
        "avg_duration_seconds": round(avg_duration, 2),
        "avg_total_tokens": round(avg_tokens, 1),
        "json_parse_success_rate": round(json_success_rate, 2),
        "twitter_length_compliance_rate": round(twitter_compliance_rate, 2),
        "verification_discard_rate": round(discard_rate, 2),
    }
class FeedbackRequest(BaseModel):
    message_id: str
    approved: bool
    timestamp: str

@app.post("/api/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    with open("approval_log.jsonl", "a", encoding="utf-8") as f:
        f.write(request.model_dump_json() + "\n")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)