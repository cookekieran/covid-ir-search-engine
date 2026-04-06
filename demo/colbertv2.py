import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
import os
import contextlib
import warnings

warnings.filterwarnings("ignore")

DOCS_PATH = Path("docs.jsonl")
RM3_OUTPUT_PATH = Path("bm25f_rm3_output.txt")
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"

try:
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig
except ImportError:
    print("error: colbert is not installed.")
    sys.exit(1)


def load_rm3_results(rm3_path):
    """load document ids and original rm3 ranking order."""
    if not rm3_path.exists():
        print(f"error: rm3 file {rm3_path} not found.")
        return [], set()

    ordered_ids = []
    doc_ids_set = set()
    with open(rm3_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                doc_id = parts[2]
                ordered_ids.append(doc_id)
                doc_ids_set.add(doc_id)

    return ordered_ids, doc_ids_set


def load_documents(file_path, filter_ids):
    """load documents from jsonl and filter by provided ids."""
    documents = {}
    if not file_path.exists():
        print(f"error: document file {file_path} not found.")
        return documents

    with open(file_path, "r") as f:
        for line in tqdm(f, desc="loading documents"):
            try:
                record = json.loads(line)
                doc_id = str(record.get("id") or record.get("docid") or record.get("sha", ""))
                if doc_id in filter_ids:
                    title = record.get("title", "").strip()
                    abstract = record.get("abstract", "").strip()
                    documents[doc_id] = f"{title} {abstract}".strip()
                if len(documents) == len(filter_ids):
                    break
            except json.JSONDecodeError:
                continue

    return documents


def rerank_documents(checkpoint, query, doc_list, doc_texts):
    """compute colbert scores and return documents sorted by relevance."""
    texts = [doc_texts[doc_id] for doc_id in doc_list]

    with torch.no_grad():
        query_tensor = checkpoint.queryFromText([query])
        doc_pack = checkpoint.docFromText(texts, bsize=32)
        docs_tensor = doc_pack[0] if isinstance(doc_pack, tuple) else doc_pack
        query_tensor = query_tensor.to(docs_tensor.dtype)
        scores = (query_tensor @ docs_tensor.permute(0, 2, 1)).max(2).values.sum(1).cpu().numpy()

    return sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)


def display_results(ranked_docs, original_order, doc_texts, top_n=10):
    """print top results with rm3 rank changes."""
    print("\n" + "=" * 80)
    print(f"{'rank':<5} | {'doc id':<12} | {'score':<8} | {'orig rank':<10} | {'change':<8}")
    print("=" * 80)

    for idx, (doc_id, score) in enumerate(ranked_docs[:top_n], 1):
        orig_rank = original_order.index(doc_id) + 1
        rank_change = orig_rank - idx
        if rank_change > 0:
            change_str = f"+{rank_change}"
        elif rank_change < 0:
            change_str = f"{rank_change}"
        else:
            change_str = "0"

        snippet = doc_texts[doc_id][:150]
        print(f"{idx:<5} | {doc_id:<12} | {score:<8.2f} | {orig_rank:<10} | {change_str}")
        print(f"      snippet: {snippet}...\n")

    print("=" * 80)


def main():
    query = "covid in kids"
    print()
    print(f"query: {query}\n")

    # load rm3 ranking and documents
    ordered_ids, relevant_ids = load_rm3_results(RM3_OUTPUT_PATH)
    if not ordered_ids:
        return

    documents = load_documents(DOCS_PATH, relevant_ids)
    if not documents:
        return

    # load colbert checkpoint silently
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        model_checkpoint = Checkpoint(COLBERT_MODEL_NAME, colbert_config=ColBERTConfig())

    filtered_doc_ids = [doc_id for doc_id in ordered_ids if doc_id in documents]

    # compute neural rerank scores silently
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        top_ranked_docs = rerank_documents(model_checkpoint, query, filtered_doc_ids, documents)

    # display top results
    display_results(top_ranked_docs, ordered_ids, documents)


if __name__ == "__main__":
    main()