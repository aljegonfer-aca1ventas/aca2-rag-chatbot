"""Prueba local del pipeline sin servicios externos: Qdrant en memoria + cliente Gemini simulado."""
import hashlib
from types import SimpleNamespace

import numpy as np
from qdrant_client import QdrantClient, models

import qdrant_utils as qu
import rag


def vec(texto):
    h = np.frombuffer(hashlib.sha256(texto.split("|")[-1].split(":",1)[-1].strip().lower().encode()).digest() * 96, dtype=np.uint8)[:768]
    v = h.astype(float) - 127.5
    return (v / np.linalg.norm(v)).tolist()


class FakeModels:
    def __init__(self):
        self.prompts = []

    def embed_content(self, model, contents, config):
        assert model == "gemini-embedding-2" and config.output_dimensionality == 768
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec(c.parts[0].text)) for c in contents])

    def generate_content(self, model, contents, config):
        if isinstance(contents, str):  # reformulación: devuelve la última pregunta tal cual
            return SimpleNamespace(text=contents.split("Última pregunta:")[-1].strip(), usage_metadata=None)
        self.prompts.append((contents, config))
        return SimpleNamespace(text="Respuesta simulada [1].", usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5))


g = SimpleNamespace(models=FakeModels())
q = QdrantClient(":memory:")
col = "aca1_ventas"
q.create_collection(col, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
qu.asegurar_indices(q, col)

# Documento "base" protegido
q.upsert(col, points=[models.PointStruct(id=1, vector=vec("garantía"), payload={
    "chunk_id": "D2_p0_c2", "doc_id": "D2", "source_file": "02_politica_garantia.md", "doc_type": "politica_garantia",
    "simulated": True, "language": "es", "text": "garantía"})])

# 1) Pregunta idéntica al fragmento → score 1 → llama al LLM
cfg = rag.Config(llm_model="fake-flash", umbral=0.6683, reformular=False)
r = rag.responder(g, q, cfg, "garantía", [])
assert not r.sin_respaldo and r.fuentes[0]["doc_id"] == "D2" and r.texto.startswith("Respuesta"), r
contents, config = g.models.prompts[-1]
assert config.temperature == 0.2 and "FRAGMENTOS RECUPERADOS" in contents[-1].parts[0].text
print("✔ RAG con evidencia → LLM llamado, fuente D2, score", r.fuentes[0]["score"])

# 2) Pregunta sin relación → bajo umbral → NO llama al LLM
n = len(g.models.prompts)
r2 = rag.responder(g, q, cfg, "teléfonos celulares", [])
assert r2.sin_respaldo and len(g.models.prompts) == n
print("✔ Sin evidencia → mensaje de no-información, sin llamar al LLM (score", r2.fuentes[0]["score"], ")")

# 3) Historial + reformulación
cfg.reformular = True
hist = [{"role": "user", "content": "garantía"}, {"role": "assistant", "content": "x [1]"}]
r3 = rag.responder(g, q, cfg, "garantía", hist)
roles = [c.role for c in g.models.prompts[-1][0]]
assert roles == ["user", "model", "user"], roles
print("✔ Historial en el prompt:", roles)

# 4) Indexar documento nuevo (MD) y eliminarlo; el base está protegido
md = ("# Ficha técnica\n\n" + "El lector tiene pantalla de 6 pulgadas. " * 60).encode()
doc_id, nfr = qu.indexar_documento(q, g, col, "ficha.md", md, "ficha_tecnica", lote=50)
df = qu.leer_payloads(q, col)
assert (df["doc_id"] == doc_id).sum() == nfr and nfr >= 2
print(f"✔ Documento indexado {doc_id} con {nfr} fragmentos; resumen:\n", qu.resumen_documentos(df).to_string(index=False))
qu.eliminar_documento(q, col, doc_id)
assert (qu.leer_payloads(q, col)["doc_id"] == doc_id).sum() == 0
try:
    qu.eliminar_documento(q, col, "D2"); raise SystemExit("debió bloquear")
except PermissionError:
    print("✔ Documento base protegido contra eliminación")

# 5) _retry no reintenta 401
calls = {"n": 0}
def f401():
    calls["n"] += 1; raise RuntimeError("401 UNAUTHENTICATED")
try:
    rag._retry(f401)
except RuntimeError:
    assert calls["n"] == 1
print("✔ Error 401 se informa de inmediato (sin reintentos)")
print("\nTODAS LAS PRUEBAS PASARON")
