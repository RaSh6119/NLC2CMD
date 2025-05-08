import os
import json
from chromadb import PersistentClient
from time import time

persist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/chroma_db"))
utility_collection_name = "unix_commands_hf"

def get_all_ids(collection, verbose=False):
    try:
        data = collection.get(include=["embeddings", "documents", "metadatas"])
        if verbose:
            print(f"[INFO] Retrieved {len(data['ids'])} documents from collection '{collection.name}'")
        return data["ids"], data["embeddings"], data["documents"], data["metadatas"]
    except Exception as e:
        print(f"[ERROR] Failed to fetch data from collection '{collection.name}': {e}")
        return [], [], [], []

def get_top3_results_with_cmd(src, dst, batch_size=128, verbose=False):
    ids, embeddings, invocations, metadatas = get_all_ids(src, verbose=verbose)
    results = []

    total_batches = (len(embeddings) + batch_size - 1) // batch_size
    if verbose:
        print(f"[INFO] Starting top-3 utility retrieval in {total_batches} batches...")

    for i in range(0, len(embeddings), batch_size):
        start_time = time()
        batch_embs = embeddings[i:i + batch_size]
        batch_invocations = invocations[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        query_results = dst.query(query_embeddings=batch_embs, n_results=5)

        for j, meta_list in enumerate(query_results["metadatas"]):
            top_utilities = list({meta.get("utility", "") for meta in meta_list if meta.get("utility")})
            true_cmd = batch_metadatas[j].get("cmd", "")
            results.append({
                "invocation": batch_invocations[j],
                "cmd": true_cmd,
                "top_3_utilities": top_utilities[:3]
            })

        if verbose:
            print(f"[BATCH {i // batch_size + 1}/{total_batches}] Processed {len(batch_embs)} items in {time() - start_time:.2f}s")

    if verbose:
        print(f"[INFO] Completed processing {len(results)} total documents.")
    return results

if __name__ == "__main__":
    verbose = True  # Set to False to suppress logs

    client = PersistentClient(path=persist_path)
    utility_collection = client.get_or_create_collection(utility_collection_name)

    src_train = client.get_or_create_collection("cmd_train")
    src_test = client.get_or_create_collection("cmd_test")

    print("[INFO] Processing training data...")
    train_results = get_top3_results_with_cmd(src_train, utility_collection, verbose=verbose)

    print("[INFO] Processing test data...")
    test_results = get_top3_results_with_cmd(src_test, utility_collection, verbose=verbose)

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/data_with_context"))
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_top3_results.json")
    test_path = os.path.join(output_dir, "test_top3_results.json")

    with open(train_path, "w") as f:
        json.dump(train_results, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"[INFO] Files saved successfully:\n - {train_path}\n - {test_path}")
