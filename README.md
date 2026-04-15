# 📄 Document Q&A Bot (RAG + LLM)

This project is a Retrieval-Augmented Generation (RAG) based Document Question Answering system.
It allows users to ask questions from multiple PDF documents and generates accurate answers strictly based on document content using an LLM.

---

## 🚀 Features

* Load and process 4–5 documents (PDF supported)
* Chunk documents into smaller segments
* Retrieve relevant information using TF-IDF
* Generate structured answers using an LLM (OpenRouter)
* Display source references with page numbers
* Clean and user-friendly Streamlit interface

---

## 🛠️ Tech Stack

* Python 3.x
* Streamlit
* LangChain (PyPDFLoader, TextSplitter)
* Scikit-learn (TF-IDF, Cosine Similarity)
* OpenRouter API (LLM)
* dotenv

---

## 🧠 Architecture Overview (RAG Pipeline)

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

1. **Ingestion**
   Load documents from the `/data` folder using PyPDFLoader

2. **Chunking**
   Split documents into smaller chunks using RecursiveCharacterTextSplitter

3. **Embedding / Representation**
   Convert text into TF-IDF vectors

4. **Retrieval**
   Use cosine similarity to retrieve top relevant chunks

5. **Generation**
   Pass retrieved context + question to LLM (OpenRouter)

---

## ✂️ Chunking Strategy

* Used: `RecursiveCharacterTextSplitter`
* Chunk size: 500
* Overlap: 50

### Why?

* Maintains context within chunks
* Prevents loss of meaning
* Improves retrieval accuracy

---

## 📊 Embedding Model & Vector DB

* Embedding method: **TF-IDF (Scikit-learn)**
* Vector storage: **In-memory matrix (no external DB)**

### Why?

* Lightweight and fast for small datasets
* No external dependencies required
* Suitable for assignment-scale documents

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd rag-document-qa-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

| Variable           | Description                |
| ------------------ | -------------------------- |
| OPENROUTER_API_KEY | API key for OpenRouter LLM |

---

## 💡 Example Queries

* What is GST?
* What is IGST?
* What is climate change?
* What is artificial intelligence?
* How does GST impact consumers?

---

## ⚠️ Known Limitations

* TF-IDF may miss semantic meaning (no deep embeddings)
* Works best for smaller document collections
* LLM depends on retrieved context quality
* Cannot answer questions outside provided documents

---

## 📂 Project Structure

```
rag-document-qa-bot/
├── app.py
├── requirements.txt
├── README.md
├── .env (not uploaded)
└── data/
    ├── GST.pdf
    ├── climate.pdf
    ├── ai.pdf
    └── economy.pdf
```

---

## 📄 License

This project is developed for educational and internship assignment purposes.
