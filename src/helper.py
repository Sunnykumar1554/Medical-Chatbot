from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.schema import Document
import csv
import os


# custom lightweight wrapper to mimic the LangChain embeddings interface
class _SimpleHuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # load the model once; SentenceTransformer returns numpy arrays
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[list[float]]:
        # PineconeVectorStore and other LangChain utilities expect a list
        # of embedding vectors for a list of input strings.
        return self._model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        # embed a single query string
        return self._model.encode([text])[0].tolist()


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks




def download_hugging_face_embeddings():
    """Return an embedding object compatible with LangChain.

    We purposely avoid the external ``langchain-huggingface`` package to
    keep dependencies simple and prevent version mismatches.
    """
    return _SimpleHuggingFaceEmbeddings()


def load_csv_file(csv_path: str, deduplicate: bool = True) -> List[Document]:
    """Load medical Q&A data from the CSV file.

    Each row becomes a LangChain Document with structured content combining
    the question description, patient query, and doctor response.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    deduplicate : bool
        If True, drop duplicate rows based on the 'Description' column.

    Returns
    -------
    List[Document]
        List of LangChain Document objects ready for embedding.
    """
    documents: List[Document] = []
    seen_descriptions: set = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            description = (row.get("Description") or "").strip()
            patient = (row.get("Patient") or "").strip()
            doctor = (row.get("Doctor") or "").strip()

            # Skip empty rows
            if not description and not patient and not doctor:
                continue

            # Deduplicate based on Description
            if deduplicate:
                if description in seen_descriptions:
                    continue
                seen_descriptions.add(description)

            # Build structured content for better retrieval
            page_content = (
                f"Question: {description}\n"
                f"Patient: {patient}\n"
                f"Doctor: {doctor}"
            )

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": os.path.basename(csv_path),
                        "type": "qa",
                    },
                )
            )

    return documents
