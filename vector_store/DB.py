from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is used widely in Artificial intelligence", metadata={"source": "doc1"}),

    Document(page_content="Pandas is used in data analysis in Python", metadata={"source": "doc2"}),

    Document(page_content="Neural network is used in deep learning", metadata={"source": "doc3"})
]

embedding_model = MistralAIEmbeddings()

vector_store = Chroma.from_documents(
    documents= docs,
      embedding = embedding_model,
      persist_directory = "chroma_db"
      )

result = vector_store.similarity_search("what is used for data analysis?", k = 2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vector_store.as_retriever()

docs = retriver.invoke("explain deep learning")

for d in docs:
    print(d.page_content)