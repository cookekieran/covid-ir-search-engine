import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from pyserini.search.lucene import LuceneSearcher
from retrieve import search_bm25f_rm3

QUERY = "covid in kids" 
INDEX_DIR = root_dir / "index"

searcher = LuceneSearcher(str(INDEX_DIR))

print(f"Query: '{QUERY}'")

# RM3
results = search_bm25f_rm3({"demo": QUERY}, index_dir=INDEX_DIR)
print(f"\n{'='*15} RM3 RANKED RESULTS {'='*15}")

if "demo" in results and results["demo"]:
    for rank, (docid, score) in enumerate(results["demo"][:5], start=1):
        doc = searcher.doc(docid)
        if doc:
            doc_json = json.loads(doc.raw())
            title = doc_json.get('title', 'No Title Available')
            
            print(f"{rank}. [{docid}] Score: {score:.2f}")
            print(f"   Title: {title}")
            print("-" * 40)
else:
    print("No results returned by the RM3 engine.")