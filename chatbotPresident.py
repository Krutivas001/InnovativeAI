import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.header("Chatbot to get details of President")
with st.sidebar:
    st.title("Present of India Details")
    file=st.file_uploader("Upload your PDF file", type=["pdf"])
    if file is not None:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key="AddLaterasPerKey")
        vector_store = FAISS.from_texts(chunks, embeddings)
        st.success("PDF file processed successfully!")
        user_question = st.text_input("Ask a question about the document:-")
        if user_question:
            response = vector_store.similarity_search(user_question)
            st.write(response)
