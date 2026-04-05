import json
from pathlib import Path
import sys
from pyserini.search.lucene import LuceneSearcher

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from retrieve import search_bm25
from config import BM25_K1, BM25_B


QUERY = "covid in kids"
INDEX_DIR = root_dir / "index"

# hyperparameters
default_k1 = BM25_K1
default_b = BM25_B
k1=3
b=0.5

searcher = LuceneSearcher(str(INDEX_DIR))

def display_top_results(results, title):
    print(f"\n{'='*10} {title} {'='*10}")
    for rank, (docid, score) in enumerate(results[:3], start=1):
        raw_doc = searcher.doc(docid).raw()
        doc_json = json.loads(raw_doc)
        print(f"{rank}. [{docid}] Score: {score:.2f} - {doc_json.get('title')[:70]}...")

# BM25
print(f"Running Initial Retrieval for: '{QUERY}'")
default_results = search_bm25({"demo": QUERY}, index_dir=INDEX_DIR)
display_top_results(default_results["demo"], f"BM25 DEFAULT (k={default_k1}, b={default_b})")

# BM25 tuned
tuned_results = search_bm25(
    {"demo": QUERY}, 
    index_dir=INDEX_DIR,
    k1=k1, 
    b=b
)
display_top_results(tuned_results["demo"], f"BM25 TUNED (k1={k1}, b={b})")