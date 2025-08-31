📘 CA Assistant – RAG System with BART Summarization

A Chartered Accountant (CA) Assistant built with Streamlit, leveraging Retrieval-Augmented Generation (RAG), FAISS vector search, and BART summarization.

This system helps users upload CA-related documents (Taxation, GST, Audit, Accounting Standards, etc.), build a searchable knowledge base, and get precise answers or concise summaries.

🚀 Features

📂 PDF Upload & Text Extraction – powered by pdfplumber

✂️ Smart Text Chunking – using LangChain’s RecursiveCharacterTextSplitter

🔍 FAISS Vector Store – efficient semantic search & retrieval

📝 Summarization – with HuggingFace BART (facebook/bart-large-cnn)

❓ Question Answering – with HuggingFace DistilBERT (deepset/minilm-uncased-squad2)

✅ Relevance Scoring – using sentence-transformers

💻 Streamlit Web App – simple, interactive Q&A interface

⚡ Tech Stack

Streamlit → Interactive web UI

Transformers (HuggingFace) → BART (Summarization) & DistilBERT (Q&A)

LangChain → Text chunking & FAISS integration

FAISS → High-performance semantic vector search

SentenceTransformers → Relevance scoring & embeddings

scikit-learn → Cosine similarity for semantic matching

✨ With this project, CAs and finance professionals can instantly query complex financial documents and receive accurate, summarized insights.
