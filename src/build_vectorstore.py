from pathlib import Path
from sentence_transformers import SentenceTransformer
import json




ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = ROOT / "knowledge_base" / "documents.json"
VECTOR_STORE_PATH = ROOT / "knowledge_base" / "vector_store.json"

print("DOCUMENTS_PATH =", DOCUMENTS_PATH)
print("Size:", DOCUMENTS_PATH.stat().st_size)

print("Loading documents...")
with open(DOCUMENTS_PATH, 'r') as f:
    documents = json.load(f)

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Creating embeddings...")
document_embeddings = {}

for disease, docs in documents.items():
    print(f"Processing {disease}...")
    disease_embeddings = []
    for doc in docs:
        text = f"{doc['title']} {doc['content']}"
        embedding = embedding_model.encode(text)
        disease_embeddings.append(embedding.tolist())
    document_embeddings[disease] = disease_embeddings

print(f"Saving vector store to {VECTOR_STORE_PATH}...")
VECTOR_STORE_PATH.parent.mkdir(exist_ok=True, parents=True)
with open(VECTOR_STORE_PATH, 'w') as f:
    json.dump(document_embeddings, f)

print("✓ Vector store created successfully!")
print(f"Total diseases: {len(document_embeddings)}")
print(f"Total documents: {sum(len(docs) for docs in document_embeddings.values())}")