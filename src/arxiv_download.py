"""
Download arXiv PDFs for a topic, ranked by arXiv's relevance sort.
Files are prefixed with rank (001_, 002_, ...) so directory order == relevance order.

pip install arxiv
python arxiv_download.py -n 30 -o supercap_pdfs
"""
import argparse
import re
import time
from pathlib import Path
import requests

import arxiv


def slug(s: str, n: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n]

def score(r):
    return 5 * r.title.lower().count("supercapacitor") + r.summary.lower().count("supercapacitor")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--query", default="all:supercapacitor OR all:supercapacitors",
                   help="arXiv query syntax")
    p.add_argument("-n", "--max", type=int, default=20, help="number of papers")
    p.add_argument("-o", "--out", default="arxiv_supercapacitor")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=a.query,
        max_results=300,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = sorted(client.results(search), key=score, reverse=True)[: a.max]

    for rank, r in enumerate(results, start=1):
        sid = r.get_short_id().replace("/", "_")
        name = f"{rank:03d}_{sid}_{slug(r.title)}.pdf"
        if (out / name).exists():
            print(f"[skip] {name}")
            continue
        try:
            resp = requests.get(
                f"https://arxiv.org/pdf/{r.get_short_id()}",
                headers={"User-Agent": "arxiv-downloader/1.0"},
                timeout=60,
            )
            resp.raise_for_status()
            (out / name).write_bytes(resp.content)
            print(f"[ok]   {name}")
        except Exception as e:
            print(f"[fail] {name}: {e}")
        time.sleep(3)


if __name__ == "__main__":

    main()