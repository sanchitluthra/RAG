import re
from typing import List
import logfire


def _split_oversized(text: str, chunk_size: int) -> List[str]:
    """
    Break a single oversized text block into pieces ≤ chunk_size.
    Strategy: try splitting on \n first, then on sentence boundaries (. ! ?),
    and finally do a hard character cut if nothing else works.
    """
    # If it already fits, return as-is
    if len(text) <= chunk_size:
        return [text]

    # Try splitting on single newlines first
    parts = text.split("\n")
    if max(len(p) for p in parts) <= chunk_size:
        return _merge_parts(parts, chunk_size, sep="\n")

    # Try splitting on sentence boundaries (. ! ? followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if max(len(s) for s in sentences) <= chunk_size:
        return _merge_parts(sentences, chunk_size, sep=" ")

    # Hard character-level cut as last resort
    pieces = []
    for i in range(0, len(text), chunk_size):
        pieces.append(text[i : i + chunk_size])
    return pieces


def _merge_parts(parts: List[str], chunk_size: int, sep: str) -> List[str]:
    """
    Greedily merge small parts back together until they approach chunk_size.
    This avoids creating tons of tiny chunks from short lines.
    """
    chunks = []
    current = ""
    for p in parts:
        # If adding this part would exceed limit, flush current chunk
        if current and len(current) + len(sep) + len(p) > chunk_size:
            chunks.append(current)
            current = p
        else:
            current = current + sep + p if current else p
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Robust chunker with overlap.
    
    1. Split on \\n\\n (paragraph boundaries) first
    2. If any paragraph > chunk_size, sub-split it (\\n → sentence → hard cut)
    3. Merge small paragraphs together up to chunk_size
    4. Apply overlap between consecutive chunks
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        # Step 1: Split on double newlines (paragraph boundaries)
        paragraphs = text.split("\n\n")

        # Step 2: Sub-split any oversized paragraphs
        segments = []
        for p in paragraphs:
            if len(p) > chunk_size:
                segments.extend(_split_oversized(p, chunk_size))
            else:
                segments.append(p)

        # Step 3: Merge small segments into chunks up to chunk_size
        raw_chunks = _merge_parts(segments, chunk_size, sep="\n\n")

        # Step 4: Apply overlap between consecutive chunks
        if overlap > 0 and len(raw_chunks) > 1:
            overlapped = [raw_chunks[0]]
            for i in range(1, len(raw_chunks)):
                # Prepend the tail of the previous chunk as context
                prev_tail = raw_chunks[i - 1][-overlap:]
                overlapped.append(prev_tail + "\n\n" + raw_chunks[i])
            raw_chunks = overlapped

        valid_chunks = [c.strip() for c in raw_chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks

        #Each file processed independently. so diff files diff chunks
        #Then all chunks go into vector DB together—  it's the metadata attached to each chunk.