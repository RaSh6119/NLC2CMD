import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../../data/utility_descriptions.csv")

df = pd.read_csv(data_path)
df.dropna(how='all', inplace=True)

def combine_fields(row):
    parts = [str(row['Utility']).strip()]
    if pd.notna(row['TLDRText']):
        parts.append(str(row['TLDRText']).strip())
    if pd.notna(row['ManpageText']):
        parts.append(str(row['ManpageText']).strip())
    return ": ".join(parts)

df['full_text'] = df.apply(combine_fields, axis=1)
texts = df['full_text'].tolist()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)


encoded_input = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors='pt'
).to(device)

with torch.no_grad():
    model_output = model(**encoded_input)

sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings_np = sentence_embeddings.cpu().numpy()


persist_path = os.path.join(base_dir, "../../utils/chroma_db")

# client = chromadb.Client(Settings(
#     persist_directory=persist_path,
#     chroma_db_impl="duckdb+parquet"
# ))

client = PersistentClient(path=persist_path)

collection = client.get_or_create_collection(name="unix_commands_hf")

collection.add(
    ids=[str(i) for i in df.index],
    documents=texts,
    embeddings=embeddings_np.tolist(),
    metadatas=[
        {
            "utility": str(df.iloc[i]['Utility']),
            "tldr": str(df.iloc[i]['TLDRText']),
            "manpage": str(df.iloc[i]['ManpageText'])
        }
        for i in range(len(df))
    ]
)

# client.persist()
print(f"HuggingFace embeddings stored in ChromaDB at: {persist_path}")
