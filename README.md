🚀 Advanced RAG Application (Groq + FAISS + Local Embeddings)

An industry-style Retrieval-Augmented Generation (RAG) application built with Streamlit, local sentence embeddings, FAISS vector search, and Groq-powered LLaMA models.
The app enables document-based question answering with grounded, source-aware responses.

📌 Project Overview

This project demonstrates a complete end-to-end RAG pipeline, designed to showcase how modern GenAI systems retrieve relevant information from documents and generate accurate answers using Large Language Models (LLMs).

The focus of this project is system design, transparency, and performance, avoiding black-box abstractions and emphasizing real-world AI engineering practices.

✨ Key Features

📄 Upload documents in PDF, DOCX, and TXT formats
✂️ Configurable text chunking with overlap control
🧠 Local embeddings using Sentence-Transformers
⚡ FAISS vector database for fast semantic search
🤖 Groq LLM integration (LLaMA models) for low-latency inference
🔍 Top-K context retrieval with source attribution
💾 Persistent FAISS indexes (save & reload)
🖥️ Interactive Streamlit web interface
🧾 Query history with retrieved evidence
🧠 System Architecture

User Query
   ↓
Sentence-Transformer Embedding
   ↓
FAISS Vector Search (Top-K)
   ↓
Context Assembly
   ↓
Prompt Engineering
   ↓
Groq LLaMA Model
   ↓
Grounded Answer

🛠️ Tech Stack

Python

Streamlit – UI & application layer
Sentence-Transformers – Local text embeddings
FAISS – Vector similarity search
Groq API – LLaMA-based LLM inference
PyPDF2 / python-docx – Document parsing
NumPy – Vector processing

📁 Project Structure


advanced_rag_app/

├── data/
│   ├── uploads/           # Uploaded documents
│   └── faiss_indexes/     # Saved FAISS indexes
├── app.py                 # Main Streamlit application
├── requirements.txt
└── README.md

⚙️ Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/Areej56/advanced_rag_app.git
cd advanced_rag_app

2️⃣ Create a virtual environment (recommended)

python -m venv venv

source venv/bin/activate      # Windows: venv\Scripts\activate

3️⃣ Install dependencies

pip install -r requirements.txt

🔐 Environment Variables

This project uses the Groq API for LLM inference.
Set your API key securely as an environment variable.

Linux / macOS

export GROQ_API_KEY="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"

Windows

setx GROQ_API_KEY "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"

▶️ Run the Application

streamlit run app.py

Open the generated local URL in your browser.

🧪 How It Works

Upload one or more documents
Text is extracted and split into overlapping chunks
Local embeddings are generated using Sentence-Transformers
FAISS index is built and stored on disk
User submits a query
Top-K relevant chunks are retrieved
LLM generates an answer strictly based on retrieved context
Sources are displayed for transparency

🔒 Why Local Embeddings?

Cost-efficient (no per-request embedding fees)
Privacy-friendly
Faster local inference
Full control over vector indexing
Production-ready architecture

🎯 Use Cases

AI-powered document Q&A systems

Enterprise knowledge bases
Research paper analysis
Internal search assistants
GenAI learning & demonstrations

📌 Skills Demonstrated

Retrieval-Augmented Generation (RAG)
Vector databases & semantic search
FAISS indexing and persistence
Prompt engineering with grounding
LLM integration (Groq / LLaMA)
Streamlit app deployment
End-to-end AI system design

👩‍💻 Author

Areej Arslan
Machine Learning & Computer Vision Engineer
📍 Lahore, Pakistan

🔗 GitHub: https://github.com/Areej56

⭐ Final Note

This project reflects industry-level GenAI engineering practices, focusing on clarity, performance, and explainability.
If you find it useful, consider ⭐ starring the repository.
