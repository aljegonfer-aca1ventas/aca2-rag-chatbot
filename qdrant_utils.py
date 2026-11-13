"""Gestión documental sobre la colección del ACA 1: estadísticas, exploración, carga y eliminación."""
import hashlib
import io
import re
import time
import uuid

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from rag import EMBED_DIM, embed

PROTEGIDOS = {"D1", "D2", "D3", "D4", "DS"}  # corpus base del ACA 1: no se elimina desde la app


def conectar(url, api_key):
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def asegurar_indices(q, col):
    """Índices de payload para filtrar por tipo y por documento (idempotente)."""
    for campo in ("doc_type", "doc_id"):
        try:
            q.create_payload_index(col, field_name=campo, field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:  # noqa: BLE001  (ya existe o el servidor no lo requiere)
            pass


def info_coleccion(q, col):
    info = q.get_collection(col)
    vp = info.config.params.vectors
    return {"puntos": q.count(col, exact=True).count, "dimension": vp.size, "distancia": str(vp.distance),
            "estado": str(info.status)}


def leer_payloads(q, col, lote=256):
    filas, offset = [], None
    while True:
        pts, offset = q.scroll(col, limit=lote, offset=offset, with_payload=True, with_vectors=False)
        filas.extend(p.payload for p in pts)
        if offset is None:
            break
    return pd.DataFrame(filas)


def resumen_documentos(df):
    if df.empty:
        return df
    g = df.groupby(["doc_id", "source_file", "doc_type"], dropna=False)
    return (g.agg(fragmentos=("chunk_id", "count"), simulado=("simulated", "first"), idioma=("language", "first"))
             .reset_index().sort_values("doc_id"))


def _limpiar(texto):
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"^---\n.*?\n---\n", "", texto, flags=re.S)  # front matter de Markdown
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def extraer_paginas(nombre, contenido: bytes):
    """Devuelve [(pagina|None, texto)]. Soporta .pdf, .md y .txt."""
    if nombre.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(contenido))
        return [(i, _limpiar(p.extract_text() or "")) for i, p in enumerate(reader.pages, start=1)]
    return [(None, _limpiar(contenido.decode("utf-8", errors="ignore")))]


def indexar_documento(q, gclient, col, nombre, contenido: bytes, doc_type, chunk_size=1000, overlap=150,
                      lote=20, progreso=None):
    """Fragmenta, vectoriza e indexa un documento nuevo. Devuelve (doc_id, n_fragmentos)."""
    doc_id = "U" + hashlib.md5(contenido).hexdigest()[:6]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
                                              separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""])
    filas = []
    for pagina, texto in extraer_paginas(nombre, contenido):
        if len(texto) < 20:
            continue
        for i, trozo in enumerate(splitter.split_text(texto)):
            filas.append({"chunk_id": f"{doc_id}_p{pagina or 0}_c{i}", "doc_id": doc_id, "source_file": nombre,
                          "doc_type": doc_type, "simulated": False, "language": "es", "page": pagina,
                          "chunk_index": i, "product": None, "rating": None, "text": trozo.strip()})
    if not filas:
        raise ValueError("No se pudo extraer texto del documento (¿PDF escaneado sin capa de texto?).")

    vectores = []
    for s in range(0, len(filas), lote):
        vectores.extend(embed(gclient, [f["text"] for f in filas[s:s + lote]], "doc"))
        if progreso:
            progreso(min(s + lote, len(filas)) / len(filas))
        time.sleep(1)
    if any(len(v) != EMBED_DIM for v in vectores):
        raise RuntimeError("Dimensión de vector distinta a la colección")

    puntos = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, "aca1-ventas/" + f["chunk_id"])),
                                 vector=list(v), payload=f) for f, v in zip(filas, vectores)]
    q.upsert(col, points=puntos, wait=True)
    return doc_id, len(filas)


def eliminar_documento(q, col, doc_id):
    if doc_id in PROTEGIDOS:
        raise PermissionError("Los documentos base del ACA 1 no se eliminan desde la aplicación.")
    q.delete(col, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])), wait=True)
