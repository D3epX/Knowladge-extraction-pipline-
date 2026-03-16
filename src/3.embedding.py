import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from utils.tokenizer import OpenAITokenizerWrapper


# Chunk sizes still follow the tokenizer-aware limit used by the original pipeline.
MAX_TOKENS = 8191

# Local embedding model (free to run, no OpenAI quota needed).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# These absolute paths keep the script runnable as a standalone file.
PDF_PATH = "/home/dahmane/dev/Knowledge-Extraction-Pipeline/src/data/docling.pdf"
DB_PATH = "/home/dahmane/dev/Knowledge-Extraction-Pipeline/src/data/lancedb"

# Initialize a local sentence-transformers model via LanceDB registry.
# This downloads the model on first run and then reuses the local cache.
func = get_registry().get("sentence-transformers").create(
     name=EMBEDDING_MODEL,
     device="cpu",
)


def get_page_number(chunk) -> int | None:
     # Docling provenance can contain several page references for one chunk.
     # The schema stores a single page number, so we keep the first sorted page.
     page_numbers = sorted(
          {
               prov.page_no
               for item in chunk.meta.doc_items
               for prov in item.prov
          }
     )
     return page_numbers[0] if page_numbers else None


class ChunkMetadata(LanceModel):
     # Metadata is stored alongside each chunk so search results keep document context.
     file_name: str | None
     page_number: int | None
     title: str | None


class Chunks(LanceModel):
     # SourceField tells LanceDB to generate embeddings from this text column.
     text: str = func.SourceField()
     vector: Vector(func.ndims()) = func.VectorField()  # pyright: ignore[reportInvalidTypeForm]
     metadata: ChunkMetadata


# The tokenizer wrapper is still useful here because Docling's HybridChunker splits by token count.
tokenizer = OpenAITokenizerWrapper()

# Step 1: convert the source PDF into Docling's structured document representation.
converter = DocumentConverter()
result = converter.convert(PDF_PATH)

# Step 2: chunk the structured document into retrieval-sized text units.
# merge_peers=True keeps adjacent related content together when possible.
chunker = HybridChunker(
     tokenizer=tokenizer,
     max_tokens=MAX_TOKENS,
     merge_peers=True,
)

chunks = list(chunker.chunk(dl_doc=result.document))

# Step 3: connect to the local LanceDB database and recreate the target table schema.
db = lancedb.connect(DB_PATH)

table = db.create_table("docling", schema=Chunks, mode="overwrite")

# Step 4: prepare rows for insertion.
# Each row includes text and metadata; LanceDB computes vectors automatically.
processed_chunks = [
     {
          "text": chunk.text,
          "metadata": {
               "file_name": chunk.meta.origin.filename,
               "page_number": get_page_number(chunk),
               "title": chunk.meta.headings[0] if chunk.meta.headings else None,
          },
     }
     for chunk in chunks
]

# Step 5: insert rows into LanceDB so the document is ready for vector search.
table.add(processed_chunks)

# Quick inspection output for manual runs.
print(table.to_pandas())
print(table.count_rows())