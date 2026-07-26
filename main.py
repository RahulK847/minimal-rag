"""
Minimal RAG API over a small collection of PDF documents (novels).

Pipeline: PDFs -> semantic chunks -> FAISS -> retriever -> Gemini -> answer.
"""

import time
from pathlib import Path

import torch
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
You are a thoughtful reading companion who has read the provided excerpts closely.

Answer the question using ONLY the provided context. Don't invent plot details, dialogue, or character traits that aren't supported by the text.

When answering:
- Pay attention to tone, mood, and emotional undercurrents in the passages, not just plot facts. If a scene is tense, melancholic, joyful, or ambiguous, let that come through in how you describe it.
- If the question asks about a character's feelings, motivations, or relationships, ground your answer in what the text shows (their actions, dialogue, described emotions) rather than just summarizing events.
- Where it strengthens the answer, you may reference brief, specific details from the passages (a gesture, a line of description, a moment of dialogue) to show where the feeling comes from — but don't quote large blocks of text.
- Match your tone to the material: if the passage is somber, don't answer in a flat, clinical way; if it's tender or funny, let that register too.

If the answer cannot be found in the context, reply exactly:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""
)



device = "cuda" if torch.cuda.is_available() else "cpu" # use gpu for faster embedding if available 

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"batch_size": 64 if device == "cuda" else 8},
)

llm = init_chat_model(model="google_genai:gemini-3.1-flash-lite", temperature=0.2)
chain = prompt | llm | StrOutputParser()

vectorstore = None  


def build_index():
    """Load every PDF in data/, split into semantic chunks, embed, index in FAISS."""
    start_time = time.time()

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

    print("Running semantic chunking (embeds every sentence locally)...")
    splitter = SemanticChunker(embeddings)
    chunks = splitter.split_documents(docs)

    result = FAISS.from_documents(chunks, embeddings)

    elapsed = int(time.time() - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    print(f"Index built in {minutes} min {seconds} sec — {len(chunks)} chunks from {len(pdf_files)} PDF(s)")

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
