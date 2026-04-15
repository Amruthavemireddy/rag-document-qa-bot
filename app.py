import streamlit as st
import os
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load API Key# 
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.title("📄 Document Q&A Bot (RAG + OpenRouter)")



# Load Documents

def load_documents():
    documents = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
            documents.extend(loader.load())
    return documents


# Chunking

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

documents = load_documents()
chunks = split_documents(documents)

pdf_files = [file for file in os.listdir("data") if file.endswith(".pdf")]
st.write(f"📄 Documents loaded: {len(pdf_files)}")
st.write(f"✂️ Chunks created: {len(chunks)}")
st.success("✅ Documents processed successfully")

st.info("💡 Tip: Ask clear questions like 'What is CGST?'")


# LLM FUNCTION (OpenRouter)

def query_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openrouter/auto",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        return f"API Error: {response.text}"

    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except:
        return str(result)


# Q&A (RAG + LLM)

st.header("🤖 Ask a Question")

query = st.text_input("Enter your question:")

if query:
    texts = [doc.page_content for doc in chunks]

    # TF-IDF Retrieval
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts + [query])

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    top_indices = similarity[0].argsort()[-3:][::-1]

    context = " ".join([texts[i] for i in top_indices])

   
    prompt = f"""
You are an expert assistant.

Answer the question using ONLY the given context.

Rules:
- First give a clear definition (1–2 lines)
- Then explain briefly (2–3 lines)
- Include important details if present
- Do NOT add outside knowledge
- Do NOT contradict the context

Context:
{context}

Question:
{query}

Answer:
"""

    answer = query_llm(prompt)

    st.subheader("📌 Answer:")
    st.write(answer)

    st.info("ℹ️ Answer generated using OpenRouter (LLM) + document context")

    st.subheader("📄 Sources:")
    for i in top_indices:
        st.write(f"- {chunks[i].metadata['source']} (Page {chunks[i].metadata['page'] + 1})")