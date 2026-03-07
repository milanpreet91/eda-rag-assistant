# eda-rag-assistant
A RAG-based technical document Q&A system that enables natural language 
querying over EDA documentation using LangChain, ChromaDB, and Groq

## Problem It Solves
EDA tools have dense, complex documentation. This system lets you ask 
questions in plain English and get precise answers from the docs instantly.

## Architecture
- **Document Loading** — ingests EDA technical docs as text
- **Chunking** — splits into overlapping segments for better context
- **Embeddings** — converts chunks to vectors using all-MiniLM-L6-v2
- **Vector Store** — ChromaDB for semantic similarity search
- **LLM** — Groq-hosted Llama3 generates natural language answers

## Tech Stack
LangChain · ChromaDB · Groq API · HuggingFace Embeddings · Python

## Setup
1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file with: `GROQ_API_KEY=your_key_here`
5. Add your EDA docs as `.txt` files to the `data/` folder
6. Run: `python app.py`

