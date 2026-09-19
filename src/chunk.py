"""Parse PDFs and split into retrieval-sized chunks, respecting section/paragraph
boundaries so chunks don't cut mid-sentence or mid-equation."""
import fitz  # PyMuPDF
import json
import os
import re

PAPERS_DIR = "papers"
META_PATH = "metadata.json"
OUT_PATH = "chunks.jsonl"

MIN_WORDS = 120
MAX_WORDS = 400

# Common section header patterns in physics/materials papers
HEADER_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(abstract|introduction|methods?|experimental|results?"
    r"|discussion|conclusion[s]?|references|acknowledg[e]?ments?)\s*$",
    re.IGNORECASE,
)


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def split_into_paragraphs(text):
    # Normalize whitespace, split on blank lines (paragraph breaks)
    text = re.sub(r"\r\n", "\n", text)
    raw_paras = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in raw_paras if p.strip()]
    return paras


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_long_paragraph(para, max_words=MAX_WORDS):
    """If a single paragraph exceeds max_words, split it at sentence
    boundaries into sub-chunks, each within the cap."""
    words = para.split()
    if len(words) <= max_words:
        return [para]

    sentences = SENTENCE_RE.split(para)
    sub_chunks = []
    current, current_words = [], 0
    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current:
            sub_chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sent)
        current_words += sent_words
    if current:
        sub_chunks.append(" ".join(current))
    return sub_chunks


def group_into_chunks(paras):
    """Greedily merge paragraphs into chunks within [MIN_WORDS, MAX_WORDS],
    starting a new chunk at section headers or when size cap is hit. Any
    single paragraph that alone exceeds MAX_WORDS is hard-split at sentence
    boundaries first."""
    chunks = []
    current = []
    current_words = 0

    for para in paras:
        is_header = bool(HEADER_RE.match(para))
        sub_paras = split_long_paragraph(para)

        for sub in sub_paras:
            sub_words = len(sub.split())

            if is_header and current_words >= MIN_WORDS:
                chunks.append(" ".join(current))
                current, current_words = [], 0
                is_header = False  # only forces a break once, at the header itself

            if current_words + sub_words > MAX_WORDS and current_words >= MIN_WORDS:
                chunks.append(" ".join(current))
                current, current_words = [], 0

            current.append(sub)
            current_words += sub_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def main():
    with open(META_PATH) as f:
        metadata = json.load(f)

    total_chunks = 0
    with open(OUT_PATH, "w") as out_f:
        for paper in metadata:
            pdf_path = paper["pdf_path"]
            if not os.path.exists(pdf_path):
                print(f"missing: {pdf_path}")
                continue

            text = extract_text(pdf_path)
            paras = split_into_paragraphs(text)
            chunks = group_into_chunks(paras)

            for i, chunk_text in enumerate(chunks):
                record = {
                    "chunk_id": f"{paper['arxiv_id']}_{i}",
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "chunk_index": i,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                }
                out_f.write(json.dumps(record) + "\n")
                total_chunks += 1

            print(f"{paper['arxiv_id']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {total_chunks}")


if __name__ == "__main__":
    main()
