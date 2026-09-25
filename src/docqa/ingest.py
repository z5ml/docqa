from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def make_overlapping_chunks(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break

    return chunks


def embed_chunk(chunk):
    return model.encode(chunk)


if __name__ == "__main__":
    text = extract_text("data/FirstDraftEEEpdf.pdf")
    chunks = make_overlapping_chunks(text, chunk_size=500, overlap=50)
    print(f"Number of chunks: {len(chunks)}")

    embedding = embed_chunk(chunks[0])
    print(f"Embedding length: {len(embedding)}")