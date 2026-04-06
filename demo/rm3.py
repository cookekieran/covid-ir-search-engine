import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent

if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from pyserini.search.lucene import LuceneSearcher
from retrieve import search_bm25f_rm3


QUERY = "covid in kids" 
INDEX_DIR = root_dir / "index"
OUTPUT_FILE = script_dir / "bm25f_rm3_output.txt"

print()
print(f"Query: '{QUERY}'")
print()

searcher = LuceneSearcher(str(INDEX_DIR))

# RM3
results = search_bm25f_rm3({"demo": QUERY}, index_dir=str(INDEX_DIR))

if "demo" in results and results["demo"]:
    all_hits = results["demo"]
    
    # save output results
    with open(OUTPUT_FILE, "w") as f:
        for rank, (docid, score) in enumerate(all_hits[:100], start=1):
            f.write(f"1 Q0 {docid} {rank} {score:.6f} \n")
    
    print(f"Saved top {len(all_hits[:100])} candidates to: {OUTPUT_FILE}")

    # print results
    print(f"\n{'='*15} RM3 RANKED RESULTS {'='*15}")
    
    count = 0
    
    for docid, score in all_hits:
        if count >= 5: break
        
        doc = searcher.doc(docid)
        if doc:
            doc_json = json.loads(doc.raw())
            title = doc_json.get('title', 'No Title Available').strip()
            
            count += 1
            
            print(f"{count}. [{docid}] Score: {score:.2f}")
            print(f"   Title: {title}")
            print("-" * 45)
else:
    print("No results returned by the RM3 engine.")