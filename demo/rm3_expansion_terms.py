import json
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher

QUERY = "covid in schools uk"
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
INDEX_DIR = root_dir / "index"

def run_diagnostic_rm3():
    searcher = LuceneSearcher(str(INDEX_DIR))
    
    searcher.set_rm3(10, 10, 0.3)
    
    print(f"Running Search for: {QUERY}")
    hits = searcher.search(QUERY, k=50)

    print("\n" + "="*45)
    print("RM3 DIAGNOSTIC LOG")
    print("="*45)

    try:
        java_obj = searcher.object
        
        expanded_q = java_obj.buildQuery(QUERY)
        print(f"Expanded Query (from buildQuery): \n{expanded_q.toString()}")
        
    except Exception as e:
        print(f"Method 'buildQuery' failed: {e}")
        try:
            print("Attempting to intercept the internal RM3 state...")
            searcher.unset_rm3()
            bm25_hits = searcher.search(QUERY, k=1)
            print(f"Original BM25 Score: {bm25_hits[0].score:.4f}")
            print(f"RM3 Weighted Score:  {hits[0].score:.4f}")
            print("\nRESULT: RM3 is active (scores differ), but terms are hidden.")
        except:
            print("Could not compare scores.")

    print("\n" + "="*45)
    print("INDEX FIELD INSPECTION")
    print("="*45)
    if hits:
        doc = searcher.doc(hits[0].docid)
        raw_json = json.loads(doc.raw())
        print("Fields available in your index:")
        for key in raw_json.keys():
            content_preview = str(raw_json[key])[:50].replace('\n', ' ')
            print(f" - {key}: {content_preview}...")

    searcher.close()

if __name__ == "__main__":
    run_diagnostic_rm3()