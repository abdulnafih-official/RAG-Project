"""Rebuild metadata.json using canonical data from the arXiv API, looked up by
the arXiv ID embedded in each filename — far more reliable than embedded PDF
metadata, which is often just Word/LaTeX export junk (usernames, template
titles, etc.)."""
import arxiv
import json
import os
import re

PAPERS_DIR = "papers"
META_PATH = "metadata.json"

# Matches e.g. 1104.3379v1, 2301.01023v1, 2511.23005v1
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}v\d+)")


def extract_arxiv_id(filename):
    match = ARXIV_ID_RE.search(filename)
    return match.group(1) if match else None


def main():
    pdf_files = sorted(
        f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")
    )

    id_to_filename = {}
    for fname in pdf_files:
        arxiv_id = extract_arxiv_id(fname)
        if arxiv_id:
            id_to_filename[arxiv_id] = fname
        else:
            print(f"no arxiv id found in filename: {fname}")

    client = arxiv.Client()
    search = arxiv.Search(id_list=list(id_to_filename.keys()))

    metadata = []
    found_ids = set()

    for result in client.results(search):
        arxiv_id = result.get_short_id()
        # arxiv API may return id without the vN suffix in some cases —
        # match against our known ids loosely
        matched_id = arxiv_id if arxiv_id in id_to_filename else next(
            (k for k in id_to_filename if k.startswith(arxiv_id.split("v")[0])),
            None,
        )
        if matched_id is None:
            continue

        fname = id_to_filename[matched_id]
        metadata.append({
            "arxiv_id": matched_id,
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat(),
            "categories": result.categories,
            "pdf_path": os.path.join(PAPERS_DIR, fname),
        })
        found_ids.add(matched_id)
        print(f"{matched_id}: {result.title[:60]}")

    missing = set(id_to_filename.keys()) - found_ids
    if missing:
        print(f"\nWARNING: {len(missing)} ids not found via API lookup:")
        for m in missing:
            print(f"  {m} ({id_to_filename[m]})")

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTotal papers with metadata: {len(metadata)}")


if __name__ == "__main__":
    main()