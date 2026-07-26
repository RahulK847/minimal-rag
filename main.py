"""
Minimal RAG API over a small collection of PDF documents.

Pipeline: PDFs -> semantic chunks -> FAISS -> retriever -> Gemini -> answer.
"""


from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATA_DIR = Path("data")
TOP_K = 4

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
reply exactly:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""
)


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

llm = init_chat_model(model="google_genai:gemini-3.1-flash-lite", temperature=0.2)
chain = prompt | llm | StrOutputParser()

vectorstore = None  



def build_index():
    docs = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data folder.")

    print(f"Found {len(pdf_files)} PDF(s), loading...")
    for path in pdf_files:
        pages = PyPDFLoader(str(path)).load()
        for p in pages:
            p.metadata["source"] = path.name
        docs.extend(pages)
    print(f"Loaded {len(docs)} pages.")

    print("Loading/downloading embedding model (first run can take a while)...")
    print("Running semantic chunking (embeds every sentence locally)...")
    splitter = SemanticChunker(embeddings)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks. Building FAISS index...")

    result = FAISS.from_documents(chunks, embeddings)
    print("Index built successfully.")
    return result


app = FastAPI(title="Minimal RAG API")


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup():
    global vectorstore
    vectorstore = build_index()


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": vectorstore.index.ntotal if vectorstore else 0}


@app.post("/ask")
def ask(req: AskRequest):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(req.question)

    context = ""
    for doc in docs:
        context += f"[{doc.metadata['source']}]\n{doc.page_content}\n\n"

    answer = chain.invoke({"context": context, "question": req.question})

    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata["source"],
            "text": doc.page_content,
        })

    return {
        "answer": answer,
        "sources": sources,
    }