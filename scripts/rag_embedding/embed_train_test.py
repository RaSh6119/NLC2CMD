import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from chromadb import PersistentClient
import gc

model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
persist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils/chroma_db"))
json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/train_test_data.json"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

def embed_texts(texts, batch_size=4):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model(**encoded)
            embeddings = mean_pooling(output, encoded["attention_mask"]).cpu()

        all_embeddings.append(embeddings)
        del batch, encoded, output, embeddings
        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(all_embeddings, dim=0).numpy()

def store_embeddings(df, collection_name, batch_size=512):
    client = PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(name=collection_name)

    texts = df["invocation"].tolist()
    cmds = df["cmd"].tolist()
    ids = [str(i) for i in df.index]
    embeddings = embed_texts(texts, batch_size=4)

    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            metadatas=[{"cmd": cmd} for cmd in cmds[i:i+batch_size]]
        )

    print(f"Stored {len(ids)} entries to collection '{collection_name}' in batches.")

if __name__ == "__main__":
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.dropna(subset=["invocation", "cmd"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    store_embeddings(train_df, "cmd_train")
    store_embeddings(test_df, "cmd_test")

    print("Train and test embeddings successfully stored in ChromaDB.")
