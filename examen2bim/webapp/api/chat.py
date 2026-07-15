import os
import io
import json
import urllib.request
from http.server import BaseHTTPRequestHandler

import numpy as np
from openai import OpenAI
from fastembed import TextEmbedding

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

EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
MODELO_LLM = "llama-3.3-70b-versatile"


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


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            largo = int(self.headers.get("Content-Length", 0))
            cuerpo = json.loads(self.rfile.read(largo) or b"{}")
            pregunta = (cuerpo.get("pregunta") or "").strip()
            if not pregunta:
                self._responder(400, {"error": "Falta el campo 'pregunta'."})
                return

            candidatos = recuperar(pregunta, k=15)
            evidencia = rerankear(pregunta, candidatos, top_n=5)
            respuesta = generar_respuesta(pregunta, evidencia)

            self._responder(200, {
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
            })
        except Exception as e:
            self._responder(500, {"error": str(e)})

    def _responder(self, codigo, cuerpo):
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(cuerpo).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
