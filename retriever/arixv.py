# from langchain_community.retrievers import ArxivRetriever

# retriever = ArxivRetriever(
#     load_max_docs = 2,
#     load_all_available_meta=True,
# )

# docs = retriever.invoke("large language models")


# for i,docs in enumerate(docs):
#     print(f"Document {i+1}:")
#     print("Title:", docs.metadata.get("Title"))
#     print("Authors:", docs.metadata.get("Authors"))
#     print("Summary:", docs.page_content[500:])
#     print("\n")

import requests
from langchain_core.documents import Document
import xml.etree.ElementTree as ET

query = "large language models"

url = (
    "https://export.arxiv.org/api/query"
    f"?search_query=all:{query.replace(' ', '+')}"
    "&start=0&max_results=2"
)

response = requests.get(url)
response.raise_for_status()

root = ET.fromstring(response.text)

ns = {"atom": "http://www.w3.org/2005/Atom"}

docs = []

for entry in root.findall("atom:entry", ns):
    title = entry.find("atom:title", ns).text.strip()
    summary = entry.find("atom:summary", ns).text.strip()

    authors = [
        author.find("atom:name", ns).text
        for author in entry.findall("atom:author", ns)
    ]

    docs.append(
        Document(
            page_content=summary,
            metadata={
                "Title": title,
                "Authors": authors,
            },
        )
    )

for doc in docs:
    print(doc.metadata["Title"])
    print(doc.metadata["Authors"])
    print(doc.page_content)
    print("=" * 80)