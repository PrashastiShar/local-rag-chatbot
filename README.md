# Local RAG AI Document Chatbot

A fully local Retrieval-Augmented Generation (RAG) pipeline that allows users to chat with their PDF documents. This project requires no external API keys or internet connection for the AI generation, ensuring complete data privacy.

## Features
*   **Document Ingestion:** Splits PDF documents into overlapping chunks using LangChain.
*   **Vector Database:** Generates text embeddings using HuggingFace and stores them in **ChromaDB** for fast semantic search.
*   **Local LLM:** Uses **Ollama** to run Llama 3.2 model entirely on the local machine's CPU.
*   **Streaming Output:** Answers are generated word-by-word for a real-time chat experience.

## How to Run
1. Install Ollama and pull the model: `ollama pull llama3.2`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your PDF as `document.pdf`.
4. Run: `python3 app.py`
