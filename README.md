# Minimal RAG API

Small RAG project: ask questions about a set of PDFs, get answers back with sources.

Pipeline: PDFs -> semantic chunks -> FAISS -> retriever -> Gemini -> answer.

## Why I built it this way
I chose semantic chunking instead of fixed-size, recursive, or agentic chunking. Agentic chunking would have been overkill for this project, as it adds extra complexity that wasn't necessary for a PDF-based RAG system. Fixed-size and recursive chunking are simpler approaches, but text extracted from PDFs often loses its original formatting and heading structure. As a result, these methods can split sentences or even complete ideas in the middle, producing less meaningful chunks. Semantic chunking, on the other hand, groups content based on where the topic naturally changes by measuring the similarity between sentence embeddings. In my testing, this approach produced more coherent chunks and delivered better overall retrieval results.

For generating embeddings, I initially used Gemini's `gemini-embedding-001` model. However, semantic chunking requires embedding almost every sentence in a document to identify natural topic boundaries, which quickly exceeded the rate limits of the free-tier API, especially for larger PDFs. To overcome this limitation, I switched to the `sentence-transformers` library with the `BAAI/bge-base-en-v1.5` model running locally. This eliminated API rate limit issues and avoided additional API costs. I still use Gemini for answer generation, as a larger hosted language model provides better response quality than a locally hosted model for that stage of the pipeline.

Running local embeddings still has a real cost: semantic chunking on CPU is slow for anything beyond a handful of pages, since it embeds nearly every sentence one batch at a time. To make iteration practical during development, the embedding model auto-detects and uses a CUDA GPU when one's available (`device="cuda"` with a larger batch size), falling back to CPU automatically otherwise. This isn't a hard requirement — the project runs fine on CPU-only machines, just slower on the initial index build. Nothing about the API or retrieval logic depends on having a GPU.


    

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # drop your GOOGLE_API_KEY in here
uvicorn main:app --reload
```

First run will take a bit longer since it has to download the embedding model. After that it's cached.

Indexing speed depends on hardware and dataset size. In my testing on an RTX 3060 (6GB), indexing 7 PDFs (2,115 pages, 5,194 chunks) took 4 min 56 sec.

CPU-only machines will take noticeably longer for the same workload, since the same embedding work runs sequentially instead of batched on GPU. Either way, the server becomes available once indexing finishes — no manual step required. Build time is logged on every startup (`Index built in X min Y sec`), so it's easy to see the actual cost on your own hardware and dataset.

## Tradeoffs / things I know aren't ideal

- Local embeddings on CPU = slow startup. Not slow per-request, just the one-time indexing cost when the server boots. To keep the indexing time manageable during development and demonstration, I limited the dataset to two PDFs.
- FAISS index isn't saved anywhere, it's rebuilt from scratch every time the app restarts. Fine for a handful of PDFs, not great once the docs pile up.
- The prompt tells the model to only answer from the given context, but there's nothing actually verifying the answer is grounded in it beyond that instruction.

## If I had more time

Probably the first thing i will do is persist the FAISS index to disk instead of rebuilding it every restart, so only new/changed PDFs get re-embedded. Second thing would be putting together a small eval set (some questions with known answers) so I can actually tell if changing the chunk size or swapping the embedding model made things better or worse, instead of just eyeballing a couple of test questions.
