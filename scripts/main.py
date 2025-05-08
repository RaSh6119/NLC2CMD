import os
import json
import torch
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from chromadb import PersistentClient
from rag_embedding.top5_rag_search import mean_pooling, tokenizer as rag_tokenizer, model as rag_model

with open("data/train_test_data.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient="index").dropna(subset=["invocation", "cmd"])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def preprocess(example):
    input_text = example["invocation"] + " <|sep|> " + example["cmd"]
    return gpt2_tokenizer(input_text, truncation=True, padding="max_length", max_length=128)

train_dataset = Dataset.from_pandas(train_df[["invocation", "cmd"]]).map(preprocess)
test_dataset = Dataset.from_pandas(test_df[["invocation", "cmd"]]).map(preprocess)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-cmds",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="no"
)

trainer = Trainer(
    model=gpt2_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForLanguageModeling(gpt2_tokenizer, mlm=False)
)

trainer.train()

def generate_cmds(invocations, model, tokenizer, max_new_tokens=20):
    model.eval()
    inputs = tokenizer(invocations, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return [tokenizer.decode(output, skip_special_tokens=True).split("<|sep|>")[-1].strip() for output in outputs]

train_preds = generate_cmds(train_df["invocation"].tolist(), gpt2_model, gpt2_tokenizer)
test_preds = generate_cmds(test_df["invocation"].tolist(), gpt2_model, gpt2_tokenizer)

exact_match = evaluate.load("exact_match")
train_em = exact_match.compute(predictions=train_preds, references=train_df["cmd"].tolist())
test_em = exact_match.compute(predictions=test_preds, references=test_df["cmd"].tolist())

print("🔍 GPT-2 Exact Match Scores:")
print(f"Train: {train_em['exact_match']:.3f}")
print(f"Test : {test_em['exact_match']:.3f}")

def rag_top1(unified_invocations):
    client = PersistentClient(path="utils/chroma_db")
    collection = client.get_or_create_collection("unix_commands_hf")

    rag_results = []
    for i in range(0, len(unified_invocations), 2):
        batch = unified_invocations[i:i+2]
        encoded = rag_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(rag_model.device)

        with torch.no_grad():
            model_output = rag_model(**encoded)
            embeddings = mean_pooling(model_output, encoded["attention_mask"]).cpu().numpy()

        for j, emb in enumerate(embeddings):
            query_result = collection.query(query_embeddings=[emb.tolist()], n_results=1)
            cmd = query_result["metadatas"][0][0].get("utility", "N/A")
            rag_results.append(cmd)

    return rag_results

train_rag_preds = rag_top1(train_df["invocation"].tolist())
test_rag_preds = rag_top1(test_df["invocation"].tolist())

train_rag_em = exact_match.compute(predictions=train_rag_preds, references=train_df["cmd"].tolist())
test_rag_em = exact_match.compute(predictions=test_rag_preds, references=test_df["cmd"].tolist())

print("\n Final Evaluation Summary (Exact Match Scores)\n")
print(f"{'Set':<10}{'GPT-2 Only':>15}{'RAG (Top-1)':>15}")
print("-" * 40)
print(f"{'Train':<10}{train_em['exact_match']:.3f}{train_rag_em['exact_match']:>15.3f}")
print(f"{'Test':<10}{test_em['exact_match']:.3f}{test_rag_em['exact_match']:>15.3f}")
