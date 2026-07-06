from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

#loader = TextLoader(r"C:\Users\ADMIN\RAG model\document_loaders\notes.txt")
# docs = loader.load()
# #print(docs)
# print(docs[0].page_content)
# #print(len(docs))

splitter = CharacterTextSplitter(separator = "",chunk_size=10, chunk_overlap=1)
data = TextLoader(r"C:\Users\ADMIN\RAG model\document_loaders\notes1.txt")
docs = data.load()
chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()

#print(len(chunks))
#print(chunks[0].page_content)