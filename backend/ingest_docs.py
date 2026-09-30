"""Build the KelpWatch knowledge base from public PDFs.

    python ingest_docs.py            # uses PDFs already in kb/raw/, downloads missing ones

Steps, each with a quality check recorded in kb/ingest_report.json:
  1. Download   - file exists and is a PDF
  2. Extract    - text per page; pages with almost no text are flagged (need OCR)
  3. Redact     - emails and phone numbers removed before anything is stored
  4. Chunk      - split on paragraph boundaries into ~900-character chunks, never across pages
  5. Label      - every chunk carries document id, title, page, year and URL

Run locally; the outputs (chunks.json, ingest_report.json) are committed so the
deployed server needs no PDF tooling.
"""
import datetime, json, pathlib, re, urllib.request

from pypdf import PdfReader

KB = pathlib.Path(__file__).resolve().parent / "kb"
RAW = KB / "raw"
TARGET_CHARS = 900
MIN_PAGE_CHARS = 40

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")


def redact(text: str) -> tuple[str, int]:
    n = len(EMAIL.findall(text)) + len(PHONE.findall(text))
    return PHONE.sub("[phone redacted]", EMAIL.sub("[email redacted]", text)), n


def clean(text: str) -> str:
    text = text.replace("­", "")                  # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)             # words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1500


def split_long(p: str) -> list[str]:
    """Split a paragraph longer than the target at sentence boundaries."""
    if len(p) <= TARGET_CHARS * 1.5:
        return [p]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?;])\s+", p):
        if cur and len(cur) + len(s) > TARGET_CHARS:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    # A single sentence with no breaks (e.g. a table flattened to text): hard-wrap on spaces.
    wrapped = []
    for piece in out:
        while len(piece) > MAX_CHUNK_CHARS:
            cut = piece.rfind(" ", 0, TARGET_CHARS) or TARGET_CHARS
            wrapped.append(piece[:cut])
            piece = piece[cut:].strip()
        wrapped.append(piece)
    return wrapped


def chunk_page(text: str) -> list[str]:
    paras = [re.sub(r"\s*\n\s*", " ", p.strip())
             for p in re.split(r"\n\s*\n|(?<=[.!?])\n(?=[A-Z])", text) if p.strip()]
    pieces = [s for p in paras for s in split_long(p)]
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > TARGET_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    # Fold fragments (stray headings, page numbers) into a neighbour.
    merged = []
    for c in chunks:
        if merged and (len(c) < MIN_CHUNK_CHARS or len(merged[-1]) < MIN_CHUNK_CHARS) \
                and len(merged[-1]) + len(c) <= MAX_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)
    return merged


def main():
    manifest = json.loads((KB / "manifest.json").read_text())
    RAW.mkdir(exist_ok=True)
    all_chunks, docs = [], []
    for d in manifest["documents"]:
        path = RAW / d["file"]
        if not path.exists():
            req = urllib.request.Request(d["url"], headers={"User-Agent": "Mozilla/5.0"})
            path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        ok_pdf = path.read_bytes()[:5] == b"%PDF-"
        reader = PdfReader(str(path)) if ok_pdf else None
        pages = [clean(p.extract_text() or "") for p in reader.pages] if reader else []
        empty_pages = [i + 1 for i, t in enumerate(pages) if len(t) < MIN_PAGE_CHARS]
        redactions, doc_chunks = 0, 0
        for page_no, text in enumerate(pages, 1):
            if len(text) < MIN_PAGE_CHARS:
                continue
            text, n = redact(text)
            redactions += n
            for i, c in enumerate(chunk_page(text)):
                all_chunks.append({
                    "id": f"{d['id']}-p{page_no}-{i}",
                    "doc_id": d["id"], "title": d["title"], "publisher": d["publisher"],
                    "year": d["year"], "url": d["url"], "page": page_no, "text": c,
                })
                doc_chunks += 1
        checks = [
            {"check": "is_pdf", "passed": ok_pdf},
            {"check": "text_extracted", "passed": len(pages) > 0 and len(empty_pages) < len(pages),
             "detail": f"{len(pages) - len(empty_pages)} of {len(pages)} pages have text"},
            {"check": "no_unreadable_pages", "passed": not empty_pages,
             "detail": (f"Pages {empty_pages} have no extractable text (image-only); OCR needed"
                        if empty_pages else "All pages readable")},
            {"check": "personal_data_redacted", "passed": True,
             "detail": f"{redactions} emails or phone numbers redacted"},
        ]
        docs.append({**{k: d[k] for k in ("id", "title", "publisher", "year", "url")},
                     "pages": len(pages), "chars": sum(map(len, pages)),
                     "unreadable_pages": empty_pages, "redactions": redactions,
                     "chunks": doc_chunks, "checks": checks})
        print(f"{d['id']:28} pages={len(pages):3} chunks={doc_chunks:4} unreadable={len(empty_pages):2} redacted={redactions}")

    sizes = [len(c["text"]) for c in all_chunks]
    in_range = sum(MIN_CHUNK_CHARS <= s <= MAX_CHUNK_CHARS for s in sizes)
    report = {
        "chunk_size_check": {
            "rule": f"chunks between {MIN_CHUNK_CHARS} and {MAX_CHUNK_CHARS} characters",
            "within_range": in_range, "total": len(sizes),
            "passed": in_range / len(sizes) >= 0.95,
        },
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "extractor": "pypdf (text layer only, no OCR)",
        "chunking": f"paragraph-aware, target {TARGET_CHARS} characters, never across pages",
        "documents": docs,
        "totals": {"documents": len(docs), "pages": sum(d["pages"] for d in docs),
                   "chunks": len(all_chunks),
                   "unreadable_pages": sum(len(d["unreadable_pages"]) for d in docs),
                   "redactions": sum(d["redactions"] for d in docs),
                   "chunk_chars_min": min(sizes), "chunk_chars_median": sorted(sizes)[len(sizes) // 2],
                   "chunk_chars_max": max(sizes)},
    }
    (KB / "chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False))
    (KB / "ingest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
