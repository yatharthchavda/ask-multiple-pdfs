# Ask Multiple PDFs

A Streamlit app that lets you chat with one or more PDF files using retrieval-augmented generation (RAG).

It:
- Extracts text from uploaded PDFs
- Splits text into chunks
- Uses Pandas/Numpy preprocessing to analyze and filter chunk quality
- Builds a FAISS vector index with Hugging Face embeddings
- Answers questions with conversation memory
- Uses Gemini API as the primary LLM and automatically falls back to a local Hugging Face model if Gemini is unavailable

## Features

- Multi-PDF upload and processing
- Conversational Q&A over document content
- Source snippet display for retrieved context
- Automatic LLM fallback:
  - Primary: `gemini-2.0-flash` (Google Gemini)
  - Fallback: `google/flan-t5-base` (local Hugging Face pipeline)
- Streamlit session state for active chat and model status

## Project Structure

- `app.py` - Main Streamlit app and RAG/chat pipeline
- `htmlTemplates.py` - UI templates and CSS
- `requirements.txt` - Python dependencies
- `docs/` - Project assets

## Requirements

- Python 3.10+
- Internet connection for first-time model downloads
- Optional Gemini API key for cloud LLM usage

Python packages used by this project:
- See `requirements.txt` for the complete dependency list.

Main packages:
- `streamlit`
- `python-dotenv`
- `pypdf`
- `pandas`
- `numpy`
- `langchain`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-classic`
- `langchain-huggingface`
- `langchain-google-genai`
- `faiss-cpu`
- `transformers`
- `sentence-transformers`
- `torch`

## Setup

1. Clone the repo and move into it.
2. Create and activate a virtual environment.
3. Install dependencies.
4. (Optional) Add Gemini API key to `.env`.
5. Run Streamlit.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional `.env`

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

If `GOOGLE_API_KEY` is missing or Gemini initialization fails, the app will show a warning and use the local Hugging Face fallback.

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How to Use

1. Open the app.
2. Upload one or more PDFs from the sidebar.
3. Click **Process**.
4. Ask questions in the input box.
5. Use **Clear Chat** to reset the conversation.

## Notes

- Embeddings are configured to run on CPU.
- Chunks are preprocessed with Pandas/Numpy before indexing to remove empty/low-signal text.
- First run can be slower due to model downloads/caching.
- If a PDF has no extractable text (for example scanned images without OCR), processing may fail with a "No readable text" message.

## Troubleshooting

- Gemini errors (quota/auth/key):
  - Check `GOOGLE_API_KEY`.
  - Verify API access and billing in Google AI Studio.
  - App should fallback to local Hugging Face automatically.

- Local model load errors:
  - Ensure `transformers`, `torch`, and `sentence-transformers` are installed.

- FAISS install issues:
  - Use `faiss-cpu` (already listed above).

