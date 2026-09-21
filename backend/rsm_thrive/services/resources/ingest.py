"""Corpus ingestion: text -> heading-aware chunks -> embedded rows."""

import re

from django.db import transaction

from rsm_thrive.models import Document, DocumentChunk

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def chunk_text(text, max_chars=1400, overlap=200):
    sections = []  # (heading, [lines])
    current = ("", [])
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.append(current)
            current = (match.group(1).strip(), [])
        else:
            current[1].append(line)
    sections.append(current)

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        if len(body) <= max_chars:
            chunks.append({"heading": heading, "text": body})
            continue
        # split on paragraph boundaries, carrying `overlap` chars forward
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        piece = ""
        for paragraph in paragraphs:
            # If paragraph itself exceeds max_chars, hard-wrap it
            if len(paragraph) > max_chars:
                # Flush any accumulated piece first
                if piece:
                    chunks.append({"heading": heading, "text": piece[:max_chars]})
                    piece = ""
                # Hard-wrap the oversized paragraph with overlap
                pos = 0
                while pos < len(paragraph):
                    chunk_end = min(pos + max_chars, len(paragraph))
                    chunks.append({"heading": heading, "text": paragraph[pos:chunk_end]})
                    if chunk_end == len(paragraph):
                        break
                    pos = chunk_end - overlap
            else:
                # Normal paragraph handling
                if piece and len(piece) + 2 + len(paragraph) > max_chars:
                    chunks.append({"heading": heading, "text": piece[:max_chars]})
                    piece = piece[-overlap:]
                piece = (piece + "\n\n" + paragraph).strip() if piece else paragraph
        if piece:
            chunks.append({"heading": heading, "text": piece[:max_chars]})
    return chunks


def extract_pdf_text(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def embedding_text(chunk):
    """What actually gets embedded for a chunk: its heading AND its text.

    Embedding the text alone left table chunks effectively unretrievable.
    Measured 2026-08-24 against the ingested corpus: the rebuilt Rady fee table
    carried the amounts and payment deadlines correctly, but asking "what does
    the MSBA cost per quarter and when is payment due?" ranked those chunks
    outside the top FOURTEEN — a chunk whose text is `| $21,279.39 | ... |`
    shares almost no lexical signal with a question, while the heading that
    describes it ("Registration fees by quarter for ... Rady School of
    Management") was stored as metadata and never embedded at all.

    Prepending the heading is the cheap half of the fix; the keyword blend in
    services/retrieval.py is the other half. Changing this changes every
    stored vector, so re-run ingest_corpus after deploying it.
    """
    heading = (chunk.get("heading") or "").strip()
    text = chunk.get("text") or ""
    # The DOCUMENT title is deliberately NOT added, though it looks like it
    # should be: 44% of chunks carry a heading of two words or fewer and 11%
    # carry none, so "$1,571.00 Per Unit" sits under "Annual Total" with nothing
    # tying it to fees or the MSBA. Tried and measured 2026-08-31 -- the golden
    # eval went 6/9 to 5/9, losing a case that had been passing. A static
    # embedding averages its tokens, so a title on every chunk adds the same
    # vector everywhere and dilutes what actually distinguishes one chunk from
    # the next. The context problem is real; this is not the fix for it.
    return f"{heading}\n\n{text}" if heading else text


@transaction.atomic
def ingest_document(source, title, kind, destinations, text, embeddings,
                    source_url=""):
    """Delete-and-recreate the document for `source`, then re-chunk and re-embed it.

    Because this deletes and recreates rather than diffing in place, chunks get
    new pks on every re-ingest. Any ChatTurnLog.chunk_ids recorded before a
    re-ingest will dangle (referencing pks that no longer resolve to the
    original chunk). Acceptable for v1; revisit when scheduled re-ingestion
    lands.
    """
    Document.objects.filter(source=source).delete()
    doc = Document.objects.create(source=source, title=title, kind=kind,
                                  destinations=destinations,
                                  source_url=source_url or "")
    chunks = chunk_text(text)
    if chunks:
        vectors = embeddings.embed([embedding_text(c) for c in chunks])
        DocumentChunk.objects.bulk_create([
            DocumentChunk(document=doc, seq=seq, heading=c["heading"],
                          text=c["text"], embedding=vector)
            for seq, (c, vector) in enumerate(zip(chunks, vectors))
        ])
    return doc
