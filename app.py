import streamlit as st
from extractor1 import MultimodalDocumentProcessor
from datetime import datetime
import lancedb
import os
import shutil
import io
import pikepdf

# Constante para recordar la última carpeta usada
LAST_FOLDER_FILE = "last_folder_path.txt"
import aspose.pdf as ap

# Configuración (ajusta según tu entorno)
CREDENTIALS_FILE_PATH = "Copia de google_ocr.json"
PROJECT_ID = "ocrtesting-464805"
LOCATION = "us"
PROCESSOR_ID = "5d4f47f17542d9f"
KB_PATH = "personal_kb"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE_PATH

# Inicialización (solo una vez)
@st.cache_resource
def load_system():
    processor = MultimodalDocumentProcessor(
        credentials=CREDENTIALS_FILE_PATH,
        project_id=PROJECT_ID,
        location=LOCATION,
        processor_id=PROCESSOR_ID
    )
    db = lancedb.connect(f"./{KB_PATH}_lancedb")
    vectordb = processor.load_existing_knowledge_base(db)
    qa_chain = processor.create_general_qa_system(vectordb)
    return processor, qa_chain



# Inicialización de qa_chain en session_state para permitir recarga dinámica
if "qa_chain" not in st.session_state or "processor" not in st.session_state:
    processor, qa_chain = load_system()
    st.session_state["qa_chain"] = qa_chain
    st.session_state["processor"] = processor
else:
    processor = st.session_state["processor"]
    qa_chain = st.session_state["qa_chain"]


# Subida y gestión de PDFs
st.sidebar.header("Gestión de PDFs")
uploaded_files = st.sidebar.file_uploader("Selecciona uno o más archivos PDF", type=["pdf"], accept_multiple_files=True)



# Mostrar PDFs existentes y permitir eliminación
import glob
pdf_dir = "./uploaded_pdfs"
os.makedirs(pdf_dir, exist_ok=True)
pdf_files = [os.path.basename(f) for f in glob.glob(os.path.join(pdf_dir, "*.pdf"))]
st.sidebar.markdown("**PDFs almacenados:**")
if pdf_files:
    selected_to_delete = st.sidebar.multiselect("Selecciona PDF(s) para eliminar", pdf_files)
    if st.sidebar.button("Eliminar seleccionados") and selected_to_delete:
        # Eliminar archivos físicos
        for fname in selected_to_delete:
            try:
                os.remove(os.path.join(pdf_dir, fname))
            except Exception as e:
                st.sidebar.warning(f"No se pudo eliminar {fname}: {e}")
        # Eliminar de la base vectorial: recargar solo los PDFs restantes
        remaining_files = [os.path.join(pdf_dir, f) for f in pdf_files if f not in selected_to_delete]
        if remaining_files:
            new_docs = processor.process_multiple_documents(remaining_files)
            processor.create_knowledge_base(new_docs, KB_PATH)
            db = lancedb.connect(f"./{KB_PATH}_lancedb")
            vectordb = processor.load_existing_knowledge_base(db)
            qa_chain = processor.create_general_qa_system(vectordb)
            st.session_state["qa_chain"] = qa_chain
            st.sidebar.success("PDF(s) y base vectorial actualizados.")
        else:
            # Si no quedan PDFs, limpiar la base vectorial (opcional: recrear vacía)
            import shutil
            kb_dir = f"./{KB_PATH}_lancedb"
            if os.path.exists(kb_dir):
                shutil.rmtree(kb_dir)
            st.session_state["qa_chain"] = None
            st.sidebar.info("No quedan PDFs. Base vectorial eliminada.")

# --- CORRECCIÓN DE INDENTACIÓN: BLOQUE DE SUBIDA Y COMPRESIÓN DE PDFS ---
if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(pdf_dir, uploaded_file.name)
        pdf_bytes = uploaded_file.getbuffer()
        size_mb = len(pdf_bytes) / (1024 * 1024)
        if size_mb > 39:
            st.sidebar.error(f"El archivo {uploaded_file.name} excede el límite de 39MB y no será cargado. Por favor, comprímelo o divídelo antes de subirlo.")
            continue
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)
        file_paths.append(file_path)

if uploaded_files:
    st.sidebar.info("Procesando archivos subidos...")
    nuevos_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(pdf_dir, uploaded_file.name)
        # Solo procesar si el archivo no existe ya en la carpeta
        if not os.path.exists(file_path):
            pdf_bytes = uploaded_file.getbuffer()
            size_mb = len(pdf_bytes) / (1024 * 1024)
            if size_mb > 40:
                st.sidebar.warning(f"El archivo {uploaded_file.name} excede 40MB. No se cargará.")
                continue
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)
            nuevos_file_paths.append(file_path)
        else:
            st.sidebar.info(f"El archivo {uploaded_file.name} ya existe y no será reprocesado.")
    if nuevos_file_paths:
        new_docs = processor.process_multiple_documents(nuevos_file_paths)
        if new_docs:
            st.sidebar.success(f"{len(new_docs)} documento(s) nuevos procesados. Actualizando base vectorial...")
            processor.create_knowledge_base(new_docs, KB_PATH)
            db = lancedb.connect(f"./{KB_PATH}_lancedb")
            vectordb = processor.load_existing_knowledge_base(db)
            qa_chain = processor.create_general_qa_system(vectordb)
            st.session_state["qa_chain"] = qa_chain
            st.sidebar.success("Base vectorial actualizada y chat recargado. Puedes consultar los nuevos documentos.")
        else:
            st.sidebar.warning("No se pudo extraer texto de los archivos nuevos subidos.")
    else:
        st.sidebar.info("No hay archivos nuevos para procesar.")

st.set_page_config(page_title="Sistema RAG", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 SISTEMA DE RECUPERACIÓN DE INFORMACIÓN BASADO EN RETRIEVAL AUGMENTED GENERATION (RAG) ")

# Mostrar la fecha actual del sistema en la interfaz
fecha_actual = datetime.now().strftime("%Y-%m-%d")
st.info(f"Fecha actual del sistema: {fecha_actual}")
st.markdown("""
<style>
    .stChatMessage {background-color: #f0f2f6; border-radius: 10px; padding: 10px; margin-bottom: 5px; color: #000;}
    .user {background-color: #d1e7dd; color: #000;}
    .bot {background-color: #f8d7da; color: #000;}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# --- Sección de extracción estructurada de datos personales ---
st.subheader("🔎 Extracción estructurada personalizada")
pregunta_struct = st.text_input("¿Qué dato(s) deseas extraer de los documentos?", value="Extrae el domicilio completo.")
if st.button("Extraer dato(s) solicitado(s)"):
    with st.spinner("Extrayendo datos solicitados..."):
        vectordb = processor.load_existing_knowledge_base(lancedb.connect(f"./{KB_PATH}_lancedb"))
        qa_chain_struct = processor.create_data_extraction_system(vectordb)
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        pregunta = f"{pregunta_struct} (Usa la fecha actual del sistema: {fecha_actual})"
        result = qa_chain_struct.invoke({
            "query": pregunta
        })
        st.success("Resultado de la extracción:")
        st.code(result["result"], language="markdown")

# --- Chat libre ---
st.subheader("💬 Chat de consulta sobre los documentos")
user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    with st.spinner("Consultando..."):
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        pregunta_con_fecha = f"{user_input}\n(Usa la fecha actual del sistema: {fecha_actual})"
        response = qa_chain.invoke({"query": pregunta_con_fecha, "fecha_actual": fecha_actual})
        answer = response["result"]
        st.session_state.chat_history.append(("usuario", user_input))
        st.session_state.chat_history.append(("sistema", answer))

# Mostrar historial de chat con burbujas
for sender, msg in st.session_state.chat_history:
    if sender == "usuario":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)

# Opcional: mostrar fuentes consultadas de la última respuesta
if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "sistema":
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    # También refuerza la instrucción en la consulta de fuentes
    last_user_q = st.session_state.chat_history[-2][1]
    last_user_q_con_fecha = f"{last_user_q}\n(Usa la fecha actual del sistema: {fecha_actual})"
    last_response = qa_chain.invoke({"query": last_user_q_con_fecha, "fecha_actual": fecha_actual})
    fuentes = last_response.get("source_documents", [])
    if fuentes:
        with st.expander("Fuentes consultadas"):
            for i, doc in enumerate(fuentes, 1):
                st.write(f"[{i}] {doc.metadata.get('source', 'Desconocido')}")
