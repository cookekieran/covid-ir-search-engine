import json
from pathlib import Path
import sys
from pyserini.search.lucene import LuceneSearcher

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from retrieve import search_bm25f
from config import BM25F_K1, BM25F_B, BM25F_WEIGHTS

QUERY = "covid in kids"
INDEX_DIR = root_dir / "index"
searcher = LuceneSearcher(str(INDEX_DIR))

# Tuned Hyperparameters for comparison
tuned_k1 = 1.5
tuned_b = 0.6
tuned_weights = {
    "title": 5.0,
    "abstract": 2.0, 
    "contents": 1.0
}

def display_results(results, title):
    print(f"\n{'='*5} {title} {'='*5}")
    for rank, (docid, score) in enumerate(results[:3], start=1):
        raw_doc = searcher.doc(docid).raw()
        doc_json = json.loads(raw_doc)
        print(f"{rank}. [{docid}] Score: {score:.2f}")
        print(f"   Title: {doc_json.get('title')[:75]}...")

# BM25F
print(f"Querying: '{QUERY}'")
res_default = search_bm25f(
    {"demo": QUERY}, 
    index_dir=INDEX_DIR
)
display_results(res_default["demo"], f"BM25F DEFAULT (k={BM25F_K1}, b ={BM25F_B}, Weights: {BM25F_WEIGHTS})")

# BM25F tuned
res_tuned = search_bm25f(
    {"demo": QUERY}, 
    index_dir=INDEX_DIR, 
    field_weights=tuned_weights,
    k1=tuned_k1,
    b=tuned_b
)
display_results(res_tuned["demo"], f"BM25F TUNED (Title Boost: {tuned_weights['title']})")