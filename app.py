import streamlit as st
import json
import pdfplumber
import re
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="CA Assistant (RAG + BART)", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    cleaned_text = re.sub(r"\s+", " ", page_text)
                    cleaned_text = re.sub(r"[^\x00-\x7F]+", "", cleaned_text)
                    text += cleaned_text + "\n"
        except Exception as e:
            st.error(f"Failed to process {getattr(pdf,'name','<unknown>')}: {e}")
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    db.save_local("faiss_index")
    with open("ca_chunks.json", "w") as f:
        json.dump([{"chunk": c} for c in text_chunks], f, indent=2)
    return db

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/minilm-uncased-squad2")

summarizer = load_summarizer()
sentence_model = load_sentence_model()
embeddings = load_embeddings()
qa_model = load_qa_model()

def user_input(user_question):
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        st.error("⚠️ No FAISS index found. Please upload & process documents first.")
        return

    coarse_docs = db.similarity_search(user_question, k=8)
    if not coarse_docs:
        st.info("No relevant chunks retrieved.")
        return

    q_emb = sentence_model.encode([user_question])[0]
    doc_texts = [d.page_content for d in coarse_docs]
    doc_embs = sentence_model.encode(doc_texts)
    sims = cosine_similarity([q_emb], doc_embs)[0]

    candidates = sorted(
        [{"text": t, "sim": float(s)} for t, s in zip(doc_texts, sims)],
        key=lambda x: x["sim"],
        reverse=True
    )

    threshold = 0.28
    selected = [c for c in candidates if c["sim"] >= threshold][:3]

    if not selected:
        st.warning("❌ Please ask a question related to the uploaded PDF(s).")
        return

    context = "\n\n".join([c["text"] for c in selected])
    if len(context) > 3000:
        context = context[:3000]

    try:
        result = qa_model(question=user_question, context=context)
        answer = result.get("answer", "").strip()
        score = result.get("score", 0.0)
    except Exception as e:
        st.error(f"QA model error: {e}")
        return

    if not answer or len(answer) < 6 or score < 0.15:
        summary = summarizer(context, max_length=120000, min_length=40, do_sample=False)[0]["summary_text"]
        st.subheader("Answer")
        st.write(summary)
    else:
        st.subheader("Answer")
        st.write(answer)
        st.caption(f"Confidence: {score:.2f}")

    st.subheader("Retrieved Context (Top Matches)")
    for i, c in enumerate(selected):
        with st.expander(f"Chunk {i+1} — sim={c['sim']:.3f}"):
            st.write(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))

def main():
    st.title("📚 CA Assistant – RAG + Summarization")
    st.markdown("Upload CA-related documents (Income Tax, GST, Audit, Ind-AS, etc.) and ask questions!")

    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.header("📂 Upload Documents")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    if chunks:
                        get_vector_store(chunks)
                        st.session_state.processed = True
                        st.success(f"✅ Indexed {len(chunks)} chunks from {len(pdf_docs)} document(s).")
                    else:
                        st.error("No chunks could be created.")
            else:
                st.warning("Please upload at least one PDF.")

        st.divider()
        if st.button("Add Sample Knowledge"):
            sample_text = """
            Direct Tax: A tax that is levied directly on an individual or organization's income or wealth.
            Examples include Income Tax and Corporate Tax.
            
            Indirect Tax: A tax that is collected by an intermediary from the person who bears the ultimate
            economic burden of the tax. GST is an example of an indirect tax.
            
            Income Tax Act 1961: Governs taxation of income in India. 
            Basic exemption limit for individuals below 60 years is Rs. 2.5 lakh.
            
            Companies Act 2013: Mandates statutory audit if paid-up capital > 10 lakh or turnover > 1 crore.
            """
            chunks = get_text_chunks(sample_text)
            if chunks:
                get_vector_store(chunks)
                st.session_state.processed = True
                st.success("✅ Sample knowledge base created!")

    if st.session_state.processed:
        user_q = st.text_input("💬 Ask a question about CA topics:", 
                               placeholder="e.g., What is direct tax?")
        if user_q:
            user_input(user_q)
    else:
        st.info("📌 Upload and process CA-related PDFs from the sidebar to get started.")

if __name__ == "__main__":
    main()
