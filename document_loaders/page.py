from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.langchain.com/oss/python/integrations/document_loaders/index#document-loader-integrations"

data = WebBaseLoader(url)

docs = data.load()

# print(len(docs))
print(docs[0].page_content)
