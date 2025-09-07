# Importación de librerías necesarias
import os
from pathlib import Path
from typing import List
from datetime import datetime
from langchain_community.vectorstores import LanceDB
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import lancedb
import re
from dataclasses import dataclass

# Importaciones específicas para Google Cloud Document AI
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

# Datos personales
@dataclass
class PersonalData:
    nombre: str = ""
    rfc: str = ""
    curp: str = ""
    edad: int = 0
    fecha_nacimiento: str = ""

    def to_dict(self):
        return {
            "nombre": self.nombre,
            "rfc": self.rfc,
            "curp": self.curp,
            "edad": self.edad
            #"fecha_nacimiento": self.fecha_nacimiento
        }

    def is_complete(self):
        return bool(self.nombre and self.rfc and self.curp and self.edad > 0)

class MultimodalDocumentProcessor:
    def __init__(self, credentials: str, output_dir: str = "./extracted_texts", project_id: str = None, location: str = None, processor_id: str = None):
        self.credentials = credentials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Configuración de Document AI
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        if self.project_id and self.location and self.processor_id:
            opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
            self.docai_client = documentai.DocumentProcessorServiceClient(client_options=opts)
            self.processor_name = self.docai_client.processor_path(self.project_id, self.location, self.processor_id)
        else:
            self.docai_client = None
            self.processor_name = None
            print("Advertencia: No se han proporcionado todos los parámetros de configuración de Document AI (project_id, location, processor_id). La carga de PDF por Document AI no estará disponible.")

    def _process_pdf_with_document_ai(self, file_path: str) -> str:
        """Procesa un PDF usando Google Cloud Document AI y devuelve el texto extraído."""
        if not self.docai_client or not self.processor_name:
            raise ValueError("Document AI no está configurado. Asegúrate de proporcionar project_id, location y processor_id al inicializar MultimodalDocumentProcessor.")

        try:
            with open(file_path, "rb") as f:
                document_content = f.read()

            request = {
                "name": self.processor_name,
                "raw_document": {"content": document_content, "mime_type": "application/pdf"},
            }
            result = self.docai_client.process_document(request=request)
            document = result.document
            return document.text
        except Exception as e:
            print(f"Error al procesar el documento {file_path} con Document AI: {e}")
            return ""

    def process_single_document(self, file_path: str) -> Document:
        """Procesa un único documento y lo convierte en un objeto Document."""
        # Se asume que solo se procesan PDFs con Document AI aquí
        if file_path.lower().endswith(".pdf"):
            text_content = self._process_pdf_with_document_ai(file_path)
            if text_content:
                return Document(page_content=text_content, metadata={"source": file_path, "type": "pdf"})
        else:
            # Para otros tipos de documentos, puedes añadir lógica aquí o simplemente ignorarlos
            print(f"Advertencia: No se puede procesar el archivo {file_path}. Solo se admiten PDFs con Document AI en esta configuración.")
        return None

    def process_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """Procesa una lista de documentos."""
        documents = []
        for file_path in file_paths:
            doc = self.process_single_document(file_path)
            if doc:
                documents.append(doc)
        return documents

    def create_knowledge_base(self, documents: List[Document], db_name="kb"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        # --- LIMPIEZA DE METADATOS ---
        allowed_keys = {"source", "type"}  # Solo los campos que usaste originalmente
        for doc in chunks:
            cleaned_metadata = {}
            for k, v in doc.metadata.items():
                if k in allowed_keys:
                    cleaned_metadata[k] = str(v)
            # Asegura que 'type' siempre esté presente
            if "type" not in cleaned_metadata:
                cleaned_metadata["type"] = "texto"
            doc.metadata = cleaned_metadata
        # --- FIN LIMPIEZA ---
        db = lancedb.connect(f"./{db_name}_lancedb")
        vectordb = LanceDB.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            connection=db,
            table_name="documents"
        )
        return vectordb

    def create_data_extraction_system(self, vectordb):
        # El prompt ahora exige explícitamente usar la fecha_actual proporcionada
        template = (
            "IMPORTANTE: Responde ÚNICAMENTE con los datos que se te solicitan en la pregunta, sin agregar ningún otro campo ni información adicional. y si no sabes la respuesta responde que no lo sabes.\n"
            "Si se solicita la edad, usa la fecha que se te proporciona en la pregunta como referencia para todos los cálculos de edad o fechas. "
            "Ignora cualquier fecha interna que tengas o que asumas por tu entrenamiento. "
            "Si se pregunta el nombre del titular del INE, responde solo con el nombre completo del titular del INE sin agregar ningún otro dato.\n"
            "Si se solicitan las fuentes consultadas, responde con las fuentes consultadas en formato [fuente1, fuente2, ...]. "
            "Si se pide el nombre del titular de uno o varios documentos, o a quien pertenece el documento responde solo con el nombre completo de el titular no de las demás personas de ese documento. "
            "Si solo se pide el domicilio, responde solo el domicilio. Si solo se pide el CURP, responde solo el CURP. No incluyas ningún campo que no se haya solicitado explícitamente.\n"
            "{context}\n"
            "Pregunta: {question}\n"
            "Respuesta:"
        )
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        retriever = vectordb.as_retriever(search_kwargs={"k": 8})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return qa_chain

    def create_general_qa_system(self, vectordb):
        """Crea un sistema de QA más general para consultas interactivas."""
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        # Un prompt más abierto que no restringe la extracción a campos específicos
        template = """Usa el siguiente contexto para responder a la pregunta.
Si no sabes la respuesta, di que no sabes, no intentes inventar una respuesta.
Si se solicita la edad, CALCÚLALA usando únicamente la fecha del sistema que se te proporciona en la pregunta, ignorando cualquier otra fecha que encuentres en los documentos o que asumas por tu entrenamiento. No muestres la edad a menos que se pida explícitamente.
Responde únicamente lo que se te pregunte, sin agregar información adicional.

IMPORTANTE: Si la pregunta solicita listar todos los documentos de una persona, responde SIEMPRE con una lista de todos los documentos encontrados para esa persona, aunque sean varios. Si hay más de un documento, enuméralos todos en la respuesta.

MUY IMPORTANTE: Si la información solicitada no está explícitamente en los documentos para la persona mencionada, responde: "No sé" o "No hay información disponible". Nunca inventes ni mezcles datos de otras personas.

{context}

Pregunta: {question}
Respuesta útil:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        retriever = vectordb.as_retriever(search_kwargs={"k": 20})
        general_qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return general_qa_chain

    def extract_personal_data(self, qa_chain, query: str) -> str:
        response = qa_chain.invoke({"query": query})
        result = response["result"]
        print(f"Respuesta del LLM:\n{result}")
        return result
    
    def load_existing_knowledge_base(self, db, table_name="documents"): #
        """Carga una base de conocimientos LanceDB ya existente""" #
        from langchain_community.vectorstores import LanceDB #
        return LanceDB(connection=db, embedding=self.embeddings, table_name=table_name) #

    def _parse_llm_response(self, text: str) -> PersonalData:
        data = PersonalData()
        try:
            data.nombre = re.search(r'NOMBRE:\s*(.*)', text).group(1).strip()
            data.rfc = re.search(r'RFC:\s*([A-Z0-9]{13})', text).group(1)
            data.curp = re.search(r'CURP:\s*([A-Z0-9]{18})', text).group(1)
            data.edad = int(re.search(r'EDAD:\s*(\d+)', text).group(1))
        except Exception as e:
            print(f"Error parseando respuesta LLM: {e}")
        return data