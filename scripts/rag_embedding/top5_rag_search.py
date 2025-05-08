import os
import json
from chromadb import PersistentClient

persist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "E:\CSCI_5832_NLP\NLC2CMD\NLC2CMD\utils\chroma_db\chroma.sqlite3"))
utility_collection = "unix_commands_hf"

def get_top3_results_with_cmd(source_collection):
    client = PersistentClient(path=persist_path)
    
    src = client.get_or_create_collection(source_collection)
    dst = client.get_or_create_collection(utility_collection)
    
    num_docs = 0
    try:
        while True:
            _ = src.get(ids=[str(num_docs)])
            num_docs += 1
    except Exception:
        pass

    ids = [str(i) for i in range(num_docs)]
    data = src.get(ids=ids)

    embeddings = data["embeddings"]
    invocations = data["documents"]
    metadatas = data["metadatas"]

    results = []

    for i, emb in enumerate(embeddings):
        query_result = dst.query(query_embeddings=[emb], n_results=3)
        top_utilities = [entry.get("utility", "") for entry in query_result["metadatas"][0]]
        true_cmd = metadatas[i].get("cmd", "")
        print(i)
        results.append({
            "invocation": invocations[i],
            "cmd": true_cmd,
            "top_3_utilities": top_utilities
        })

    return results

if __name__ == "__main__":
    train_results = get_top3_results_with_cmd("cmd_train")
    test_results = get_top3_results_with_cmd("cmd_test")

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/data_with_context"))
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train_top3_results.json"), "w") as f:
        json.dump(train_results, f, indent=2)

    with open(os.path.join(output_dir, "test_top3_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    print("Files saved")
