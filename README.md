# ACA 2 · Chatbot RAG de atención comercial (Pilar Ventas)

Aplicación **Streamlit** que completa el sistema RAG del ACA 1:
**Qdrant Cloud** (colección `aca1_ventas`, 527 fragmentos, 768 dim, coseno) + **Gemini Embedding 2** + **LLM de Gemini**.

**Equipo:** Hugo Alberto Benitez Morelo · Alvaro de Jesús González Fernández · Andres David Torres Villalobos
**Docente:** Hamilton Rivera Flor · Especialización en Inteligencia Artificial – CUN

> Las políticas de la tienda *NovaLectura* son **documentos académicos simulados**. Las reseñas provienen del dataset público *Consumer Reviews of Amazon Products* (Datafiniti, Kaggle).

## Funcionalidades
- **Chat** con historial conversacional, reformulación de preguntas de seguimiento y **fuentes citadas** [n] con su similitud.
- **Mitigación de alucinaciones:** umbral de similitud (0,6683, calibrado en el ACA 1) antes de llamar al LLM, *system prompt* estricto y temperatura baja.
- **Gestión documental:** estado de la colección, documentos y fragmentos indexados, consulta semántica directa, **carga** de nuevos PDF/MD/TXT y **eliminación** de documentos agregados (el corpus base está protegido).
- **Panel de configuración:** modelo LLM, temperatura, máx. tokens, Top-k, umbral y filtro por tipo de documento.
- Tiempos por etapa (embedding, búsqueda, generación) y tokens consumidos.

## Arquitectura
```
Usuario ─► Streamlit (chat + gestión documental)
   1. Reformulación con historial (opcional, LLM)
   2. Embedding de la consulta ─► gemini-embedding-2 (768, formato consulta)
   3. Top-k ─► Qdrant Cloud · aca1_ventas (Cosine)
   4. ¿Mejor score < umbral? ─► "No hay información suficiente" (sin LLM)
   5. Prompt: system + fragmentos [n] + historial + pregunta ─► LLM Gemini
   6. Respuesta con citas + fuentes visibles
```

## Estructura
| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz Streamlit (chat, gestión documental, configuración) |
| `rag.py` | Pipeline RAG: embeddings, recuperación, umbral, prompt y generación |
| `qdrant_utils.py` | Estadísticas, exploración, carga y eliminación de documentos |
| `test_local.py` | Pruebas sin servicios externos (Qdrant en memoria + Gemini simulado) |
| `requirements.txt` | Dependencias con versiones probadas |
| `.streamlit/secrets.toml.example` | Plantilla de credenciales (sin valores) |

## Ejecución local
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # completar con las claves reales
streamlit run app.py
```
Pruebas sin servicios externos: `python test_local.py`

## Despliegue en Streamlit Community Cloud
1. share.streamlit.io → **Create app** → repositorio, rama `main`, archivo `app.py`.
2. **Advanced settings** → Python **3.13** → en **Secrets** pegar el contenido de `secrets.toml.example` con los valores reales.
3. **Deploy**.

## Seguridad
- Ninguna credencial en el código ni en el repositorio: se leen de `st.secrets` o de `.env` (ambos excluidos por `.gitignore`).
- Acceso a la app protegido con contraseña (`APP_PASSWORD`) para proteger la cuota gratuita.
- Los errores 401/403 (clave inválida) se informan de inmediato; solo se reintentan errores transitorios (429/5xx).

## Referencias
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS, 33*, 9459–9474.
- Google. (2026). *Gemini API documentation*. https://ai.google.dev/gemini-api/docs
- Qdrant. (s. f.). *Qdrant documentation*. https://qdrant.tech/documentation/
- Streamlit Inc. (2024). *Streamlit documentation*. https://docs.streamlit.io/
