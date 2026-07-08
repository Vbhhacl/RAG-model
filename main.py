from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import lambda_stmt



load_dotenv()  # Load environment variables from .env file

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4 ,"fetch_k": 10, "lambda_mult": 0.5}
)

llm = ChatMistralAI(model="mistral-small-2506")

#data = PyPDFLoader(r"C:\Users\ADMIN\RAG model\document_loaders\GRU.pdf")
# data = PyPDFLoader(r"C:\Users\ADMIN\RAG model\document_loaders\deep-learning-material.pdf")
# docs = data.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# chunks = splitter.split_documents(docs)

#prompt template
# template = ChatPromptTemplate.from_messages([
#     ("system", "you are an AI that summarizes the text"),
#     ("human", "{data}")
# ])
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)
print("rag system created successfully")

print("press 0 to exit")

while True:
    query = input("YOU : ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_with_context = prompt.format_prompt(context=context, question=query)

    result = llm.invoke(prompt_with_context.to_messages())
    print("Answer:", result.content)

# model = ChatMistralAI(model ="mistral-small-2506")

# #prompt = template.format_messages(data=docs[0].page_content)
# prompt = template.format_messages(data=docs)
# #result = model.invoke("Hello")
# result = model.invoke(prompt)
# print(result.content)



