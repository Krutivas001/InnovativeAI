import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("📄 Chat with your PDF using HuggingFace + LangChain")

# File uploader
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Load and split PDF
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    # HuggingFace Embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector DB
    db = FAISS.from_documents(documents, embeddings_model)

    # Define retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=generator)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Chat interface
    st.subheader("💬 Ask questions about your PDF")
    user_query = st.text_input("Enter your question:")

    if user_query:
        response = qa_chain.run(user_query)
        st.write("**Answer:**", response)