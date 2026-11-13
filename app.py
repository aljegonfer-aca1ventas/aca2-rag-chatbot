"""ACA 2 – Chatbot RAG de atención comercial (Pilar Ventas) · Streamlit + Qdrant Cloud + Gemini."""
import os

import pandas as pd
import streamlit as st
from google import genai

import qdrant_utils as qu
import rag

st.set_page_config(page_title="Asistente NovaLectura · RAG", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------- Credenciales
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def secreto(nombre, defecto=None):
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:  # noqa: BLE001  (sin secrets.toml en ejecución local)
        pass
    return os.getenv(nombre, defecto)


GEMINI_API_KEY = secreto("GEMINI_API_KEY")
QDRANT_URL = secreto("QDRANT_URL")
QDRANT_API_KEY = secreto("QDRANT_API_KEY")
APP_PASSWORD = secreto("APP_PASSWORD")
COLLECTION = secreto("QDRANT_COLLECTION", "aca1_ventas")
LLM_DEFECTO = secreto("LLM_MODEL", "")

faltan = [n for n, v in [("GEMINI_API_KEY", GEMINI_API_KEY), ("QDRANT_URL", QDRANT_URL),
                         ("QDRANT_API_KEY", QDRANT_API_KEY)] if not v]
if faltan:
    st.error(f"Faltan credenciales: {', '.join(faltan)}. Configúralas en los Secrets de Streamlit o en un archivo .env.")
    st.stop()

# ---------------------------------------------------------------- Acceso con contraseña
if APP_PASSWORD and not st.session_state.get("autenticado"):
    st.title("🛒 Asistente NovaLectura")
    clave = st.text_input("Contraseña de acceso", type="password")
    if st.button("Ingresar"):
        if clave == APP_PASSWORD:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    st.caption("Acceso restringido para proteger la cuota gratuita de las APIs.")
    st.stop()


# ---------------------------------------------------------------- Clientes (una vez por sesión del servidor)
@st.cache_resource
def clientes():
    g = genai.Client(api_key=GEMINI_API_KEY)
    q = qu.conectar(QDRANT_URL, QDRANT_API_KEY)
    qu.asegurar_indices(q, COLLECTION)
    return g, q


@st.cache_data(ttl=3600, show_spinner=False)
def modelos_generativos():
    """Modelos disponibles para la clave configurada que admiten generateContent."""
    g, _ = clientes()
    nombres = []
    for m in g.models.list():
        acciones = getattr(m, "supported_actions", None) or []
        if "generateContent" in acciones:
            nombres.append(m.name.replace("models/", ""))
    return sorted(nombres)


@st.cache_data(ttl=120, show_spinner=False)
def payloads():
    _, q = clientes()
    return qu.leer_payloads(q, COLLECTION)


try:
    gclient, qclient = clientes()
except Exception as e:  # noqa: BLE001
    st.error(f"No fue posible conectar con los servicios: {e}")
    st.info("Si el error menciona Qdrant, revisa en cloud.qdrant.io que el clúster esté activo (no suspendido).")
    st.stop()

# ---------------------------------------------------------------- Barra lateral: configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    try:
        lista = modelos_generativos()
    except Exception as e:  # noqa: BLE001
        lista = []
        st.warning(f"No se pudo listar modelos: {e}")
    if LLM_DEFECTO and LLM_DEFECTO not in lista:
        lista = [LLM_DEFECTO] + lista
    if not lista:
        st.error("No hay modelos disponibles. Define LLM_MODEL en los Secrets.")
        st.stop()
    idx = lista.index(LLM_DEFECTO) if LLM_DEFECTO in lista else next(
        (i for i, n in enumerate(lista) if "flash" in n and "lite" not in n), 0)
    modelo = st.selectbox("Modelo LLM (Gemini API)", lista, index=idx)
    temperatura = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05,
                            help="Baja = respuestas más fieles y repetibles; alta = más creativas.")
    max_tokens = st.slider("Máx. tokens de salida", 256, 4096, 1024, 128)
    top_k = st.slider("Fragmentos recuperados (Top-k)", 1, 10, 5)
    umbral = st.slider("Umbral de similitud", 0.50, 0.90, 0.6683, 0.0001, format="%.4f",
                       help="Calibrado en el ACA 1. Si el mejor fragmento queda por debajo, no se llama al LLM.")
    tipos = {"Todos": None, "Devoluciones": "politica_devoluciones", "Garantía": "politica_garantia",
             "Términos y SLA": "terminos_servicio_sla", "Reseñas de clientes": "review"}
    filtro = st.selectbox("Buscar en", list(tipos))
    reformular = st.toggle("Usar historial para preguntas de seguimiento", value=True)
    if st.button("🗑️ Nueva conversación"):
        st.session_state.mensajes = []
        st.rerun()
    st.caption(f"Colección: `{COLLECTION}` · Embeddings: `{rag.EMBED_MODEL}` ({rag.EMBED_DIM} dim)")

cfg = rag.Config(llm_model=modelo, collection=COLLECTION, top_k=top_k, umbral=umbral, temperatura=temperatura,
                 max_tokens=max_tokens, reformular=reformular, doc_type=tipos[filtro])

st.title("🛒 Asistente NovaLectura")
st.caption("Chatbot RAG de atención comercial · Qdrant Cloud + Gemini · ACA 2 Computación en la Nube (CUN). "
           "Las políticas de NovaLectura son documentos académicos simulados.")

tab_chat, tab_docs, tab_info = st.tabs(["💬 Chat", "📚 Gestión documental", "ℹ️ Acerca de"])


def mostrar_fuentes(res):
    if not res.get("fuentes"):
        return
    with st.expander(f"📎 Fuentes ({len(res['fuentes'])}) · tiempos y detalles"):
        if res.get("consulta_usada") and res["consulta_usada"] != res.get("pregunta"):
            st.caption(f"Consulta reformulada para la búsqueda: *{res['consulta_usada']}*")
        for f in res["fuentes"]:
            tag = " · simulado" if f.get("simulated") else ""
            st.markdown(f"**[{f['n']}] {f['source_file']}** · `{f['doc_type']}`{tag} · similitud **{f['score']:.4f}**")
            st.caption(f["text"][:400] + ("…" if len(f["text"]) > 400 else ""))
        t = res.get("tiempos", {})
        if t:
            st.caption(" · ".join(f"{k}: {v:.2f} s" for k, v in t.items()))
        if res.get("tokens"):
            st.caption(f"Tokens → entrada: {res['tokens'].get('entrada')} · salida: {res['tokens'].get('salida')}")


# ---------------------------------------------------------------- Pestaña Chat
with tab_chat:
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
    if not st.session_state.mensajes:
        st.info("Ejemplos: ¿Cuántos días tengo para retractarme de una compra? · ¿Qué cubre la garantía? · "
                "¿Qué opinan los clientes de la batería?")
    for m in st.session_state.mensajes:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                mostrar_fuentes(m.get("meta", {}))

    pregunta = st.chat_input("Escribe tu pregunta sobre devoluciones, garantías, envíos o productos…")
    if pregunta:
        historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes]
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los documentos y generando la respuesta…"):
                try:
                    r = rag.responder(gclient, qclient, cfg, pregunta, historial)
                    meta = {"pregunta": pregunta, "consulta_usada": r.consulta_usada, "fuentes": r.fuentes,
                            "tiempos": r.tiempos, "tokens": r.tokens, "sin_respaldo": r.sin_respaldo}
                    texto = ("⚠️ " if r.sin_respaldo else "") + r.texto
                except Exception as e:  # noqa: BLE001
                    meta, texto = {}, f"❌ Ocurrió un error al consultar los servicios: {e}"
            st.markdown(texto)
            mostrar_fuentes(meta)
        st.session_state.mensajes.append({"role": "assistant", "content": texto, "meta": meta})

# ---------------------------------------------------------------- Pestaña Gestión documental
with tab_docs:
    try:
        info = qu.info_coleccion(qclient, COLLECTION)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fragmentos (puntos)", info["puntos"])
        c2.metric("Dimensión", info["dimension"])
        c3.metric("Distancia", info["distancia"].split(".")[-1].title())
        c4.metric("Estado", info["estado"].split(".")[-1].title())
        df = payloads()
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo leer la colección: {e}")
        df = pd.DataFrame()

    if not df.empty:
        st.subheader("Documentos indexados")
        resumen = qu.resumen_documentos(df)
        st.dataframe(resumen, width="stretch", hide_index=True)
        st.bar_chart(df["doc_type"].value_counts())

        st.subheader("Explorar fragmentos")
        doc_sel = st.selectbox("Documento", resumen["doc_id"] + " · " + resumen["source_file"])
        doc_id = doc_sel.split(" · ")[0]
        cols = [c for c in ["chunk_id", "page", "product", "rating", "text"] if c in df.columns]
        st.dataframe(df[df["doc_id"] == doc_id][cols].head(200), width="stretch", hide_index=True)

    st.subheader("Consulta semántica directa (sin LLM)")
    consulta = st.text_input("Pregunta para ver qué fragmentos recupera Qdrant")
    if consulta:
        try:
            v = rag.embed(gclient, [consulta], "query")[0]
            hits = rag.recuperar(qclient, COLLECTION, v, top_k, tipos[filtro])
            st.dataframe(pd.DataFrame(hits)[["n", "score", "doc_id", "doc_type", "source_file", "chunk_id", "text"]],
                         width="stretch", hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.subheader("Agregar un documento al índice")
    st.caption("Formatos: PDF con texto, Markdown o TXT. Se fragmenta (1.000/150), se vectoriza con "
               f"{rag.EMBED_MODEL} y se indexa en Qdrant. Consume cuota de la API.")
    archivo = st.file_uploader("Documento", type=["pdf", "md", "txt"])
    tipo_nuevo = st.selectbox("Tipo de documento", ["ficha_tecnica", "politica_devoluciones", "politica_garantia",
                                                    "terminos_servicio_sla", "manual_usuario", "otro"])
    if archivo and st.button("📥 Indexar documento"):
        barra = st.progress(0.0, text="Vectorizando…")
        try:
            nuevo_id, n = qu.indexar_documento(qclient, gclient, COLLECTION, archivo.name, archivo.getvalue(),
                                               tipo_nuevo, progreso=lambda p: barra.progress(p, text="Vectorizando…"))
            payloads.clear()
            st.success(f"Documento {archivo.name} indexado como {nuevo_id} ({n} fragmentos).")
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo indexar: {e}")

    if not df.empty:
        cargados = sorted(d for d in df["doc_id"].dropna().unique() if d not in qu.PROTEGIDOS)
        if cargados:
            st.subheader("Eliminar un documento agregado")
            borrar = st.selectbox("Documento agregado", cargados)
            if st.button("🗑️ Eliminar del índice"):
                try:
                    qu.eliminar_documento(qclient, COLLECTION, borrar)
                    payloads.clear()
                    st.success(f"{borrar} eliminado.")
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))

# ---------------------------------------------------------------- Pestaña Acerca de
with tab_info:
    st.markdown("""
**Arquitectura RAG**

1. La pregunta (y, si aplica, el historial) se reformula como consulta autónoma.
2. Se vectoriza con **Gemini Embedding 2** (768 dimensiones, formato consulta).
3. **Qdrant Cloud** devuelve los Top-k fragmentos más similares (coseno) de la colección del ACA 1.
4. Si el mejor fragmento no supera el **umbral calibrado (0,6683)**, el sistema responde que no hay información (no llama al LLM).
5. El **LLM de Gemini** redacta la respuesta solo con esos fragmentos, citándolos como [n].

**Mitigación de alucinaciones:** umbral previo, *system prompt* estricto, temperatura baja y fuentes visibles.

**Equipo:** Hugo Alberto Benitez Morelo · Alvaro de Jesús González Fernández · Andres David Torres Villalobos
· Docente: Hamilton Rivera Flor · Especialización en IA – CUN.
""")
