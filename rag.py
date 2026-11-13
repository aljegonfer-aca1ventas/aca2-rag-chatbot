"""Pipeline RAG: embedding de la consulta, recuperación en Qdrant, control por umbral y generación con Gemini."""
import time
from dataclasses import dataclass, field

from google.genai import types
from qdrant_client import models

# Mismos parámetros del ACA 1 (deben coincidir con los vectores ya indexados)
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
QUERY_PREFIX = "task: question answering | query: "
DOC_TEMPLATE = "title: none | text: {text}"

NO_INFO_MSG = ("No encuentro información suficiente en los documentos disponibles para responder "
               "esa pregunta con respaldo. Puedes reformularla o consultar con un asesor.")

SYSTEM_PROMPT = """Eres el asistente virtual de atención al cliente de NovaLectura, una tienda en línea
(caso académico; sus políticas son documentos simulados).

Reglas obligatorias:
1. Responde ÚNICAMENTE con la información de los FRAGMENTOS que se te entregan. No uses conocimiento externo.
2. Cita la fuente de cada afirmación con su número entre corchetes, por ejemplo [1] o [2][3].
3. Si los fragmentos no contienen la respuesta, di exactamente: "No encuentro esa información en los
   documentos disponibles." No inventes plazos, valores, condiciones ni productos.
4. Las reseñas (tipo review) son opiniones de clientes, no especificaciones oficiales: preséntalas como opiniones.
5. Responde en español, de forma clara y breve (máximo 6 oraciones o una lista corta)."""


@dataclass
class Config:
    llm_model: str
    collection: str = "aca1_ventas"
    top_k: int = 5
    umbral: float = 0.6683
    temperatura: float = 0.2
    max_tokens: int = 1024
    reformular: bool = True
    doc_type: str | None = None
    turnos_historial: int = 4


@dataclass
class Resultado:
    texto: str
    fuentes: list
    sin_respaldo: bool
    consulta_usada: str
    tiempos: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)


def _retry(fn, intentos=4, espera=10):
    """Reintenta solo errores transitorios (429 cuota, 500/503). Un 401/403 se informa de inmediato."""
    for i in range(intentos):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transitorio = any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE"))
            if not transitorio or i == intentos - 1:
                raise
            time.sleep(espera * (2 ** i))


def embed(gclient, textos, tipo="query"):
    """Vectoriza textos. tipo='query' para preguntas, 'doc' para fragmentos de documentos."""
    fmt = (lambda t: QUERY_PREFIX + t) if tipo == "query" else (lambda t: DOC_TEMPLATE.format(text=t))
    contents = [types.Content(parts=[types.Part(text=fmt(t))]) for t in textos]  # 1 Content = 1 vector
    resp = _retry(lambda: gclient.models.embed_content(
        model=EMBED_MODEL, contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM)))
    vecs = [e.values for e in resp.embeddings]
    if len(vecs) != len(textos):
        raise RuntimeError(f"Se esperaban {len(textos)} vectores y llegaron {len(vecs)}")
    return vecs


def recuperar(qclient, collection, vector, top_k=5, doc_type=None):
    flt = None
    if doc_type:
        flt = models.Filter(must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))])
    pts = qclient.query_points(collection, query=vector, limit=top_k, query_filter=flt, with_payload=True).points
    return [{
        "n": i + 1, "score": round(float(p.score), 4),
        "doc_id": p.payload.get("doc_id"), "doc_type": p.payload.get("doc_type"),
        "source_file": p.payload.get("source_file"), "chunk_id": p.payload.get("chunk_id"),
        "product": p.payload.get("product"), "simulated": p.payload.get("simulated"),
        "text": p.payload.get("text", ""),
    } for i, p in enumerate(pts)]


def construir_contexto(fuentes):
    bloques = []
    for f in fuentes:
        extra = f" · Producto: {f['product']}" if f.get("product") else ""
        bloques.append(f"[{f['n']}] Fuente: {f['source_file']} · Tipo: {f['doc_type']}{extra}\n{f['text']}")
    return "\n\n".join(bloques)


def _historial_a_contents(historial, turnos):
    """historial: lista de {'role': 'user'|'assistant', 'content': str}. Solo los últimos turnos."""
    out = []
    for m in historial[-2 * turnos:]:
        role = "user" if m["role"] == "user" else "model"
        out.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return out


def reformular_pregunta(gclient, cfg, pregunta, historial):
    """Convierte una pregunta de seguimiento ('¿y si es una tableta?') en una pregunta autónoma para la búsqueda."""
    if not historial:
        return pregunta
    previo = "\n".join(f"{'Cliente' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
                       for m in historial[-2 * cfg.turnos_historial:])
    prompt = ("Reescribe la ÚLTIMA pregunta del cliente como una pregunta autónoma y completa, en español, "
              "usando la conversación previa solo para resolver referencias. Devuelve solo la pregunta.\n\n"
              f"Conversación previa:\n{previo}\n\nÚltima pregunta: {pregunta}")
    resp = _retry(lambda: gclient.models.generate_content(
        model=cfg.llm_model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256)))
    texto = (resp.text or "").strip()
    return texto or pregunta


def responder(gclient, qclient, cfg: Config, pregunta: str, historial: list) -> Resultado:
    t = {}
    t0 = time.perf_counter()
    consulta = pregunta
    if cfg.reformular and historial:
        consulta = reformular_pregunta(gclient, cfg, pregunta, historial)
        t["reformulación"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    vector = embed(gclient, [consulta], "query")[0]
    t["embedding"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    fuentes = recuperar(qclient, cfg.collection, vector, cfg.top_k, cfg.doc_type)
    t["búsqueda Qdrant"] = time.perf_counter() - t2

    mejor = fuentes[0]["score"] if fuentes else 0.0
    if mejor < cfg.umbral:  # Capa 1 anti-alucinación: sin evidencia suficiente no se llama al LLM
        t["total"] = time.perf_counter() - t0
        return Resultado(NO_INFO_MSG, fuentes, True, consulta, t)

    usuario = (f"FRAGMENTOS RECUPERADOS:\n{construir_contexto(fuentes)}\n\n"
               f"PREGUNTA DEL CLIENTE: {pregunta}\n"
               "Responde siguiendo las reglas y citando los fragmentos con [n].")
    contents = _historial_a_contents(historial, cfg.turnos_historial)
    contents.append(types.Content(role="user", parts=[types.Part(text=usuario)]))

    t3 = time.perf_counter()
    resp = _retry(lambda: gclient.models.generate_content(
        model=cfg.llm_model, contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                           temperature=cfg.temperatura,
                                           max_output_tokens=cfg.max_tokens)))
    t["generación LLM"] = time.perf_counter() - t3
    t["total"] = time.perf_counter() - t0

    texto = (resp.text or "").strip() or "El modelo no devolvió texto. Intenta de nuevo o aumenta el máximo de tokens."
    um = getattr(resp, "usage_metadata", None)
    tokens = {"entrada": getattr(um, "prompt_token_count", None), "salida": getattr(um, "candidates_token_count", None)} if um else {}
    return Resultado(texto, fuentes, False, consulta, t, tokens)
