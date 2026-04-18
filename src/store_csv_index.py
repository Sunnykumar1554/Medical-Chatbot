"""
store_csv_index.py
──────────────────
Load the ai-medical-chatbot.csv file, deduplicate, split into chunks,
and upsert embeddings into the existing Pinecone 'medical-chatbot' index.

NOTE: The full CSV has ~228K unique Q&A pairs which creates ~733K chunks.
Uploading all of them would take 13+ hours. Instead, we upload the first
MAX_DOCS unique entries which keeps the upload fast while still providing
a rich medical Q&A knowledge base.
"""

from dotenv import load_dotenv
import os
import time
from src.helper import load_csv_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ── Configuration ───────────────────────────────────────────────────────
MAX_DOCS = 5000        # Number of unique Q&A docs to upload (set to None for all)
BATCH_SIZE = 200       # Chunks per Pinecone batch upload


def clean_text(text: str) -> str:
    """Remove non-ASCII characters that cause encoding issues with Pinecone."""
    return text.encode("ascii", errors="ignore").decode("ascii")


# ── 1. Load & deduplicate CSV ───────────────────────────────────────────
csv_path = os.path.join("data", "ai-medical-chatbot.csv")
print(f"[1/5] Loading CSV from: {csv_path}")

csv_docs = load_csv_file(csv_path, deduplicate=True)
total_unique = len(csv_docs)
print(f"       Found {total_unique} unique Q&A documents (after deduplication)")

# Limit to MAX_DOCS if set
if MAX_DOCS and len(csv_docs) > MAX_DOCS:
    csv_docs = csv_docs[:MAX_DOCS]
    print(f"       Using first {MAX_DOCS} documents for upload")

# ── 2. Clean text and split into chunks ─────────────────────────────────
print("[2/5] Cleaning text and splitting into chunks...")
for doc in csv_docs:
    doc.page_content = clean_text(doc.page_content)

text_chunks = text_split(csv_docs)
print(f"       Created {len(text_chunks)} text chunks from {len(csv_docs)} documents")

# ── 3. Initialize embeddings ────────────────────────────────────────────
print("[3/5] Loading embedding model...")
embeddings = download_hugging_face_embeddings()

# ── 4. Ensure Pinecone index exists ─────────────────────────────────────
index_name = "medical-chatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(index_name):
    print(f"[4/5] Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"[4/5] Pinecone index '{index_name}' already exists")

# ── 5. Batch upload to Pinecone ─────────────────────────────────────────
total_chunks = len(text_chunks)
total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

print(f"[5/5] Uploading {total_chunks} chunks in {total_batches} batches...")

failed_batches = 0
uploaded_chunks = 0
start_time = time.time()

for i in range(0, total_chunks, BATCH_SIZE):
    batch = text_chunks[i : i + BATCH_SIZE]
    batch_num = (i // BATCH_SIZE) + 1

    try:
        PineconeVectorStore.from_documents(
            documents=batch,
            index_name=index_name,
            embedding=embeddings,
        )
        uploaded_chunks += len(batch)
        elapsed = time.time() - start_time
        rate = uploaded_chunks / elapsed if elapsed > 0 else 0
        eta = (total_chunks - uploaded_chunks) / rate if rate > 0 else 0
        print(f"  ✓ Batch {batch_num}/{total_batches} "
              f"({uploaded_chunks}/{total_chunks}) "
              f"[{elapsed:.0f}s, ~{eta/60:.1f}m left]")
    except Exception as e:
        print(f"  ⚠ Batch {batch_num} failed: {str(e)[:80]}")
        time.sleep(5)
        try:
            PineconeVectorStore.from_documents(
                documents=batch,
                index_name=index_name,
                embedding=embeddings,
            )
            uploaded_chunks += len(batch)
            print(f"  ✓ Batch {batch_num} retry OK")
        except Exception as e2:
            failed_batches += 1
            print(f"  ✗ Batch {batch_num} failed permanently: {str(e2)[:80]}")

elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✓ CSV data upload complete!")
print(f"  Q&A docs uploaded:  {len(csv_docs)} (of {total_unique} total)")
print(f"  Chunks uploaded:    {uploaded_chunks}")
print(f"  Failed batches:     {failed_batches}")
print(f"  Time:               {elapsed_total/60:.1f} minutes")
print(f"  Pinecone index:     {index_name}")
print(f"{'='*60}")
