# RAG Chat (PDF Q&A) using Chroma + Mistral

This project is a simple Retrieval-Augmented Generation (RAG) system built with:

- **LangChain**
- **Chroma** vector database (persisted locally)
- **Mistral** models for embeddings + chat (via `langchain-mistralai`)
- **PDF ingestion** using `PyPDFLoader`
- **Streamlit UI** for uploading a PDF and asking questions
- **CLI** RAG loop for interactive querying

---

## Project Structure

- `app.py` – Streamlit app (upload PDF → build Chroma DB → chat)
- `main.py` – CLI loop (loads the persisted Chroma DB → retrieves context → answers)
- `create_db.py` – example script to create a Chroma DB from a PDF (hard-coded path)
- `vector_store/DB.py` – example/testing vector-store interactions
- `document_loaders/` – PDF/text loader experiments
- `retriever/` – retrieval strategy experiments (MMR, MultiQuery, etc.)
- `chroma_db/` – persisted Chroma database (created by the scripts)
- `uploads/` – where uploaded PDFs are stored by Streamlit

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables (.env)

This project uses `python-dotenv`.

Create a `.env` file in the project root with values required by `langchain-mistralai` (typically your Mistral API key).

> The exact variable names depend on the `langchain-mistralai` integration you use.

At minimum, you should set the API credentials required for:
- `MistralAIEmbeddings`
- `ChatMistralAI`

---

## Build the Vector Database

### Option 1 (recommended): Streamlit flow

`app.py` builds the Chroma database dynamically for the uploaded PDF.

### Option 2: CLI/example script

`create_db.py` demonstrates how to:
- load a PDF
- split it into chunks
- embed chunks
- store them in `chroma_db/`

⚠️ `create_db.py` currently contains a hard-coded Windows path to a sample PDF. Edit it to match your file if you use it.

---

## Run the Streamlit App (PDF chat)

```bash
streamlit run app.py
```

### How to use
1. Upload a **PDF** in the UI
2. Click **Create Knowledge Base**
3. Ask a question
4. The app retrieves relevant chunks from Chroma using **MMR** and answers using ONLY the retrieved context

---

## Run the CLI RAG Loop

`main.py` expects an existing persisted Chroma DB in `./chroma_db`.

```bash
python main.py
```

### Usage
- Enter a query at the prompt
- Type `0` to exit

---

## Retrieval Strategy

In both `app.py` and `main.py`, retrieval is configured using MMR:

- `search_type="mmr"`
- `k` – number of chunks returned
- `fetch_k` – number of candidates fetched before diversification
- `lambda_mult` – tradeoff between relevance and diversity

---

## Notes / Limitations

- The prompt instructs the model to **use ONLY the provided context**.
- If the answer is not present in retrieved chunks, the model should respond:
  - **`I could not find the answer in the document.`**
- `chroma_db/` is persisted locally; if you upload a new PDF, `app.py` deletes and recreates the directory.

---

## Dependencies

See `requirements.txt` for the full list.

---

## Quick Checklist

1. Add proper keys in `.env`
2. Install requirements
3. Run `streamlit run app.py`
4. Optionally run `python main.py` after a DB has been created

