from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
import faiss
import numpy as np

load_dotenv()  # must run before OpenAI() is created
client = OpenAI()  # reads OPENAI_API_KEY from the environment

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
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


def embed(texts, batch_size=100):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return np.array(vectors, dtype="float32")


def build_index(chunks):
    embeddings = embed(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, index, chunks, k=5):
    query_embedding = embed([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


def answer(query, index, chunks, k=5):
    context = "\n\n---\n\n".join(retrieve(query, index, chunks, k))

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    text = extract_text("data/FirstDraftEEEpdf.pdf")
    chunks = make_overlapping_chunks(text, chunk_size=300, overlap=50)
    print(f"Number of chunks: {len(chunks)}")

    index = build_index(chunks)
    print(f"Total vectors in index: {index.ntotal}")

    question = "What is the main topic of the document?"
    print(answer(question, index, chunks))