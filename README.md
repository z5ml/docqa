# docqa

Ask questions about a PDF. The script extracts the text, splits it into overlapping chunks, embeds them with OpenAI, retrieves the most relevant chunks with FAISS, and passes them to an LLM to generate the answer.

## How it works

1. **Extract**: `pypdf` reads the text from the PDF.
2. **Chunk**: the text is split into overlapping word-based chunks (500 words, 50 overlap by default).
3. **Embed**: each chunk is embedded with `text-embedding-3-small`.
4. **Index**: the vectors are stored in a FAISS `IndexFlatL2` index.
5. **Retrieve**: the question is embedded and the 5 nearest chunks are returned.
6. **Answer**: `gpt-4o-mini` answers using only the retrieved chunks, and says so if the answer isn't in them.

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

```bash
git clone https://github.com/z5ml/docqa.git
cd docqa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

`.env` is gitignored. Never commit your key.

## Usage

1. Create a `data/` folder and put your PDF in it.
2. Update the PDF path and the question in `src/docqa/ingest.py`.
3. Run it from the project root:

```bash
python src/docqa/ingest.py
```

The script prints the number of chunks, the retrieved chunks with their distances, and the generated answer.

## Configuration

| Setting | Location | Default |
|---|---|---|
| Chunk size (words) | `make_overlapping_chunks(...)` call | 500 |
| Overlap (words) | `make_overlapping_chunks(...)` call | 50 |
| Neighbours retrieved (`k`) | `k` variable | 5 |
| Embedding model | `EMBED_MODEL` | `text-embedding-3-small` |
| Chat model | `CHAT_MODEL` | `gpt-4o-mini` |

## Limitations

- The index is rebuilt on every run, so each run makes embedding API calls.
- Scanned PDFs without a text layer return no text. OCR is not supported.
- Chunking is by word count, not by sentence or section.
- The PDF path and question are hardcoded.
