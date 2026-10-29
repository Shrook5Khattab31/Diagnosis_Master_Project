import os, ollama, chromadb
from pypdf import PdfReader

client = chromadb.PersistentClient(path="db")
col = client.get_or_create_collection("books")

def chunks(text, size=400, overlap=60):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i + size])

for fname in os.listdir("books"):
    if not fname.endswith(".pdf"):
        continue
    reader = PdfReader(f"books/{fname}")
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for j, chunk in enumerate(chunks(text)):
            if len(chunk.split()) < 30:   # skip near-empty chunks
                continue
            emb = ollama.embed(model="nomic-embed-text", input=chunk)["embeddings"][0]
            col.add(
                ids=[f"{fname}-{page_no}-{j}"],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"book": fname, "page": page_no}],
            )
    print("done:", fname)