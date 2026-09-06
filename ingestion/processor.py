import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from service.config import settings

from service.retrival.embedding import embed_texts, get_embedding_dim
from ingestion.loaders.pdf import parse_pdf
from ingestion.loaders.html_loader import parse_html
from ingestion.loaders.text import parse_text
from ingestion.chunking.splitters import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
# connnecting qdrant with  python
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

#convert data to json 
#But Qdrant does not need this JSON file to perform vector search.
# The JSON is basically your local processed-data/metadata backup.
def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        # Skip if already successfully processed (JSON is saved only after successful indexing)
        check_path = os.path.join(PROCESSED_DATA_DIR, source_type, f"{filename}.json")
        if os.path.exists(check_path):
            logfire.info(f"⏭️ Skipping {filename} — already indexed.")
            return

        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]## check .pdf or .html 
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Embed and index in Qdrant
            #Each chunk becomes a Qdrant point.
            """
            Qdrant Point
            │
            ├── ID
            │
            ├── Vector
            │
            └── Payload
                ├── text
                ├── source
                └── source_type
            """
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    #one record/point that Qdrant can store.
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]
                #Insert these points into Qdrant,
                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")

            # 4. Save processed metadata locally — ONLY after successful indexing
            # This acts as the "done" marker so re-runs skip this file
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


# scan files one by one then then run pure ingestion pipeline
def process_directory(dir_path: str, source_type: str, max_files: int = None):
    """Process every file in a directory. If max_files is set, only process that many."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        total = len(files)
        if max_files and len(files) > max_files:
            files = files[:max_files]
            logfire.info(f"Found {total} files in {dir_path} — limiting to {max_files}.")
        else:
            logfire.info(f"Found {total} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


# Default: noisy data limited to 6 files to avoid API quota issues
NOISY_FILE_LIMIT = 6

def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False, limit: int = None):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion
    wipe is used for if database have someting it delete all and start fresh 
    when I want to completely rebuild my vector database.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            #This creates the container/database collection where your vectors will live.
            #Qdrant needs to know the similarity metric when the collection is created
            # even though you're not searching yet.
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    #"When vector similarity/search happens, use cosine as the distance metric."
                    #not now extact 
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            max_f = limit if limit else (NOISY_FILE_LIMIT if source_type == "noisy" else None)
            process_directory(base_dir, source_type, max_files=max_f)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                # For noisy folders, limit files to avoid API quota issues
                max_f = limit if limit else (NOISY_FILE_LIMIT if source_type == "noisy" else None)
                process_directory(os.path.join(base_dir, subdir), source_type, max_files=max_f)


if __name__ == "__main__":
    # Usage:
    #   python -m ingestion.processor DATA --wipe
    #   python -m ingestion.processor DATA/true_data true
    #   python -m ingestion.processor DATA --wipe --limit 10
    wipe_requested = "--wipe" in sys.argv

    # Parse --limit N flag
    file_limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            file_limit = int(sys.argv[idx + 1])

    clean_args = [a for a in sys.argv if a not in ("--wipe", "--limit") and not a.isdigit() or a == sys.argv[0]]
    # Simpler: just grab positional args
    positional = [a for a in sys.argv[1:] if not a.startswith("--") and a not in ([str(file_limit)] if file_limit else [])]

    target_dir = positional[0] if len(positional) > 0 else "DATA"
    explicit_type = positional[1] if len(positional) > 1 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested, limit=file_limit)
    logfire.info("Ingestion job completed.")