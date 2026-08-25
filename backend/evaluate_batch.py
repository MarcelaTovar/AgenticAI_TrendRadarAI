import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from langchain_ollama import ChatOllama

ARCHIVE_PATH = Path("reports_archive.jsonl")
RESULTS_PATH = Path("evaluation_results.jsonl")

judge_llm = ChatOllama(
    model="llama3.2:3b-instruct-fp16",
    temperature=0,       # queremos consistencia, no creatividad, al evaluar
    format="json"         # modo JSON nativo de Ollama — más confiable que pedirlo solo en el prompt
)

def build_judge_prompt(topic: str, verified_data: str, executive_report: str) -> str:
    return f"""
Eres un evaluador de calidad estricto para reportes generados por IA.
Compara el REPORTE GENERADO contra los DATOS VERIFICADOS originales sobre '{topic}'.

DATOS VERIFICADOS (fuente de verdad):
{verified_data}

REPORTE GENERADO (a evaluar):
{executive_report}

Evalúa en una escala de 1 a 5 (donde 5 es excelente):

1. fidelity_score: ¿El reporte usa ÚNICAMENTE información presente en los datos
   verificados, sin inventar cifras, nombres o hechos que no aparecen ahí?
2. relevance_score: ¿El reporte se mantiene enfocado en '{topic}' sin desviarse
   a generalidades no relacionadas?
3. completeness_score: ¿El reporte cubre los puntos principales de los datos
   verificados sin omitir información importante?

Responde ÚNICAMENTE con este JSON:
{{
  "fidelity_score": <entero 1-5>,
  "relevance_score": <entero 1-5>,
  "completeness_score": <entero 1-5>,
  "hallucination_detected": <true/false>,
  "notes": "<una oración breve explicando la puntuación más baja>"
}}
"""

def evaluate_record(record: dict) -> dict:
    prompt = build_judge_prompt(
        record.get("topic", ""),
        record.get("verified_data", ""),
        record.get("executive_report", "")
    )
    start = time.perf_counter()
    try:
        response = judge_llm.invoke(prompt)
        raw = re.sub(r"^```(json)?|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        result = json.loads(raw)
        result["eval_success"] = True
    except Exception as e:
        result = {
            "fidelity_score": None,
            "relevance_score": None,
            "completeness_score": None,
            "hallucination_detected": None,
            "notes": f"Error de evaluación: {e}",
            "eval_success": False
        }

    result["topic"] = record.get("topic")
    result["source_timestamp"] = record.get("timestamp")
    result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    result["eval_duration_seconds"] = round(time.perf_counter() - start, 3)
    return result

def main():
    if not ARCHIVE_PATH.exists():
        print("No hay reports_archive.jsonl todavía. Genera algunos reportes primero.")
        return

    records = [json.loads(line) for line in ARCHIVE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"📦 {len(records)} reportes encontrados. Evaluando con modelo local...\n")

    results = []
    for i, record in enumerate(records, start=1):
        print(f"  [{i}/{len(records)}] Evaluando tema: '{record.get('topic')}'...")
        result = evaluate_record(record)
        results.append(result)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Resumen agregado en consola
    successful = [r for r in results if r["eval_success"]]
    if successful:
        avg_fidelity = sum(r["fidelity_score"] for r in successful) / len(successful)
        avg_relevance = sum(r["relevance_score"] for r in successful) / len(successful)
        avg_completeness = sum(r["completeness_score"] for r in successful) / len(successful)
        hallucinations = sum(1 for r in successful if r.get("hallucination_detected"))

        print("\n=== RESUMEN DE EVALUACIÓN ===")
        print(f"Reportes evaluados exitosamente: {len(successful)}/{len(results)}")
        print(f"Fidelidad promedio:    {avg_fidelity:.2f}/5")
        print(f"Relevancia promedio:   {avg_relevance:.2f}/5")
        print(f"Completitud promedio:  {avg_completeness:.2f}/5")
        print(f"Alucinaciones detectadas: {hallucinations}/{len(successful)}")
    else:
        print("⚠️  Ningún reporte pudo evaluarse correctamente.")

    print(f"\n✅ Resultados guardados en '{RESULTS_PATH}'")

if __name__ == "__main__":
    main()