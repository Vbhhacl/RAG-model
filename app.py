import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat With Your PDF")
st.write("Upload a PDF and ask questions.")

UPLOAD_DIR = "uploads"
DB_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)

embedding_model = MistralAIEmbeddings()

llm = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context.

If the answer cannot be found, reply:

'I could not find the answer in the document.'
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    pdf_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Create Knowledge Base"):

        with st.spinner("Processing PDF..."):

            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=DB_DIR
            )

        st.success("Knowledge Base Created!")

if os.path.exists(DB_DIR):

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    st.divider()

    question = st.text_input(
        "Ask a question"
    )

    if st.button("Ask"):

        if question:

            with st.spinner("Searching..."):

                docs = retriever.invoke(question)

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                final_prompt = prompt.format_prompt(
                    context=context,
                    question=question
                )

                result = llm.invoke(
                    final_prompt.to_messages()
                )

            st.subheader("Answer")

            st.write(result.content)

            with st.expander("Retrieved Context"):

                for i, doc in enumerate(docs):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(doc.page_content)