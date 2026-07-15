import os
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from fastembed import TextEmbedding

# El unico directorio con permiso de escritura dentro de una funcion
# serverless de Vercel es /tmp. Le decimos explicitamente a fastembed que
# use esa carpeta para descargar y cachear el modelo ONNX (por defecto
# intenta escribir en ~/.cache, que en Vercel es de solo lectura y hace
# que la funcion "crashee" en el primer arranque en frio).
CACHE_DIR = "/tmp/fastembed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# El corpus (4000 documentos) y sus embeddings pesan varios MB, demasiado
# para empaquetarlos dentro del deployment de la funcion serverless. En vez
# de eso los descargamos una sola vez (al "arrancar en frio" la funcion)
# desde el repositorio de GitHub donde vive el resto del examen, y los
# guardamos en variables globales para que las siguientes invocaciones
# (mientras la funcion siga "caliente") no vuelvan a descargarlos.
RAW_BASE = "https://raw.githubusercontent.com/danielife05/ir26a/main/examen2bim/webapp/api/data"


def _descargar(nombre):
    url = f"{RAW_BASE}/{nombre}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


CORPUS = json.loads(_descargar("corpus.json").decode("utf-8"))
EMBEDDINGS = np.load(io.BytesIO(_descargar("embeddings.npy")))

EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=CACHE_DIR)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
MODELO_LLM = "llama-3.3-70b-versatile"

INDEX_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


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
