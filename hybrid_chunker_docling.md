First: how Docling represents a document

Before chunking happens, Docling converts your PDF into a structured document tree.

Example:

Document
│
├── Title
│
├── Section 1
│     ├── Paragraph
│     ├── Paragraph
│     └── Table
│
├── Section 2
│     ├── Paragraph
│     ├── List
│     └── Paragraph

Each node contains:

node.text
node.type
node.metadata
node.children

Example node:

ParagraphNode(
   text="Machine learning models require large datasets...",
   page=4,
   section="Introduction"
)

This structure is critical because HybridChunker works on these nodes.

2️⃣ HybridChunker algorithm overview

The algorithm roughly looks like this:

document
   ↓
extract structural nodes
   ↓
merge nodes into chunks
   ↓
count tokens
   ↓
if chunk too large → split

So the algorithm has two phases.

STRUCTURE PHASE
TOKEN PHASE
3️⃣ Phase 1 — Structural chunking

HybridChunker walks through document nodes.

Pseudo-code:

for node in document_nodes:

    if node is heading:
        start new chunk

    add node text to chunk

Example document:

Heading: Introduction
Paragraph A
Paragraph B
Paragraph C

Chunk result:

Chunk 1
Introduction
Paragraph A
Paragraph B
Paragraph C

This preserves semantic context.

4️⃣ Phase 2 — Chunk merging

HybridChunker tries to merge smaller nodes into larger chunks.

Example:

Paragraph A → 120 tokens
Paragraph B → 150 tokens
Paragraph C → 180 tokens

Instead of making 3 chunks:

A
B
C

It merges:

Chunk:
A + B + C

Because total tokens:

120 + 150 + 180 = 450

Still safe.

5️⃣ Phase 3 — Token counting

Now the chunker calls your tokenizer:

tokenizer.tokenize(chunk_text)

Which internally uses tiktoken.

Example:

Chunk tokens = 620

Compare with limit:

MAX_TOKENS = 8191

Since:

620 < 8191

Chunk is accepted.

6️⃣ Phase 4 — Oversized chunk splitting

If a chunk becomes too large:

tokens > MAX_TOKENS

HybridChunker tries progressive splitting.

Step order:

1️⃣ split by sections
2️⃣ split by paragraphs
3️⃣ split by sentences
4️⃣ split by tokens

Example:

Chunk = 10,000 tokens

Split by paragraphs:

Paragraph A → 3000
Paragraph B → 3000
Paragraph C → 4000

Now we get:

Chunk 1 → A + B (6000)
Chunk 2 → C (4000)

Both valid.

7️⃣ Final chunk structure

Each chunk returned contains metadata.

Example:

Chunk
{
   text: "...",
   tokens: 620,
   page: 4,
   section: "Introduction",
   document_id: "doc123"
}

Metadata is important for retrieval later.

8️⃣ Why HybridChunker works better than naive splitters

Naive splitter (like simple LangChain):

split every 500 tokens

Result:

"Machine learning models are trained"
"on large datasets to improve"
"performance across many..."

Meaning breaks.

HybridChunker keeps:

Paragraph boundaries
Section boundaries
Table context
List context

So retrieval becomes much more accurate.

9️⃣ Internal logic summary

Actual simplified algorithm:

chunks = []

current_chunk = ""

for node in document_nodes:

    candidate = current_chunk + node.text

    if tokens(candidate) < MAX_TOKENS:
        current_chunk = candidate

    else:
        chunks.append(current_chunk)
        current_chunk = node.text

chunks.append(current_chunk)
10️⃣ Real pipeline in your system

 code is implementing this pipeline:

PDF
 ↓
DocumentConverter
 ↓
Docling structured document
 ↓
HybridChunker
 ↓
tokenizer counts tokens
 ↓
semantic chunks
 ↓
OpenAI embeddings
 ↓
vector database
11️⃣ Why HybridChunker matters for RAG

Good chunking improves:

retrieval precision
context coherence
LLM answers

Bad chunking causes:

lost context
hallucinations
irrelevant retrieval