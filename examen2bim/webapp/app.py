import os

# El unico directorio con permiso de escritura dentro de una funcion
# serverless de Vercel es /tmp. huggingface_hub (usado internamente por
# fastembed) revisa HF_HOME para varias rutas de cache/locks, sin importar
# lo que se le pase como cache_dir a TextEmbedding, asi que lo fijamos
# ANTES de importar fastembed. Ademas forzamos "modo offline": el modelo
# ya viene empaquetado con el deployment (carpeta model_cache/), asi que
# no hace falta ni conviene que intente red en cada arranque en frio.
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).parent

# El corpus y sus embeddings viajan empaquetados con el deployment (son
# parte del repositorio de git), asi que se leen directo del disco: no se
# descargan por red en cada arranque en frio.
with open(BASE_DIR / "api" / "data" / "corpus.json", encoding="utf-8") as f:
    CORPUS = json.load(f)

EMBEDDINGS = np.load(BASE_DIR / "api" / "data" / "embeddings.npy")

# El modelo de embeddings (ONNX cuantizado, ~65 MB) tambien viaja
# empaquetado en model_cache/, con la misma estructura de cache que usa
# huggingface_hub. Al estar completo localmente, la carga es instantanea
# y no depende de la red ni de escribir en disco.
EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(BASE_DIR / "model_cache"))

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
MODELO_LLM = "llama-3.3-70b-versatile"

INDEX_HTML = (BASE_DIR / "index.html").read_text(encoding="utf-8")


def recuperar(pregunta, k=15):
    vector = np.array(list(EMBED_MODEL.embed([pregunta])), dtype="float32")[0]
    similitudes = EMBEDDINGS @ vector
    mejores_posiciones = np.argsort(-similitudes)[:k]
    return [
        {**CORPUS[i], "similitud": float(similitudes[i])}
        for i in mejores_posiciones
    ]


def rerankear(pregunta, candidatos, top_n=5):
    lista = "\n".join(
        f"[{c['doc_id']}] {c['titulo']}: {c['abstract'][:300]}" for c in candidatos
    )
    prompt = f"""Eres un sistema de re-ranking para busqueda academica.
Pregunta del usuario: "{pregunta}"

Candidatos (id, titulo, resumen recortado):
{lista}

Elige los {top_n} candidatos MAS relevantes para responder la pregunta, ordenados del mas al menos relevante.
Responde solo JSON: {{"ranked_ids": ["id1", "id2", ...]}}"""

    resp = client.chat.completions.create(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        ids_ordenados = json.loads(resp.choices[0].message.content)["ranked_ids"]
    except Exception:
        ids_ordenados = [c["doc_id"] for c in candidatos[:top_n]]

    por_id = {c["doc_id"]: c for c in candidatos}
    seleccion = [por_id[i] for i in ids_ordenados if i in por_id]
    if not seleccion:
        seleccion = candidatos[:top_n]
    return seleccion[:top_n]


def generar_respuesta(pregunta, evidencia):
    contexto = "\n\n".join(
        f"[Documento {c['doc_id']}] {c['titulo']}\n{c['abstract']}" for c in evidencia
    )
    prompt_sistema = (
        "Eres un asistente que responde preguntas sobre articulos cientificos de arXiv. "
        "Responde UNICAMENTE con base en los documentos de contexto que se te dan. "
        "Cita los documentos relevantes usando su [Documento id]. "
        "Si el contexto no contiene informacion suficiente para responder con confianza, "
        "dilo explicitamente en vez de inventar una respuesta."
    )
    prompt_usuario = f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"

    resp = client.chat.completions.create(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


class PreguntaEntrada(BaseModel):
    pregunta: str


@app.post("/api/chat")
def chat(cuerpo: PreguntaEntrada):
    pregunta = (cuerpo.pregunta or "").strip()
    if not pregunta:
        return JSONResponse(status_code=400, content={"error": "Falta el campo 'pregunta'."})

    candidatos = recuperar(pregunta, k=15)
    evidencia = rerankear(pregunta, candidatos, top_n=5)
    respuesta = generar_respuesta(pregunta, evidencia)

    return {
        "respuesta": respuesta,
        "evidencia": [
            {
                "doc_id": c["doc_id"],
                "titulo": c["titulo"],
                "abstract": c["abstract"][:500],
                "categorias": c.get("categorias", ""),
                "similitud": round(c["similitud"], 4),
            }
            for c in evidencia
        ],
    }
