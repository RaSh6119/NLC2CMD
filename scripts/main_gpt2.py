import os
import json
import time
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

with open("data/train_test_data.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient="index").dropna(subset=["invocation", "cmd"])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

def preprocess(example):
    input_text = example["invocation"] + " <|sep|> " + example["cmd"]
    return gpt2_tokenizer(input_text, truncation=True, padding="max_length", max_length=128)

train_dataset = Dataset.from_pandas(train_df[["invocation", "cmd"]]).map(preprocess)
test_dataset = Dataset.from_pandas(test_df[["invocation", "cmd"]]).map(preprocess)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-cmds",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=5,
    num_train_epochs=3,
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
save_dir = "./gpt2-finetuned-cmds"
trainer.save_model(save_dir)
gpt2_tokenizer.save_pretrained(save_dir)

def generate_cmds(invocations, model, tokenizer, batch_size=8, max_new_tokens=20):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    start_time = time.time()

    for i in range(0, len(invocations), batch_size):
        batch = invocations[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = [
            tokenizer.decode(output, skip_special_tokens=True).split("<|sep|>")[-1].strip()
            for output in outputs
        ]
        predictions.extend(decoded)

        del inputs, outputs, decoded
        torch.cuda.empty_cache()

    end_time = time.time()
    return predictions, end_time - start_time

test_preds, test_gen_time = generate_cmds(test_df["invocation"].tolist(), gpt2_model, gpt2_tokenizer)

exact_match = evaluate.load("exact_match")
hf_bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def compute_all_metrics(preds, refs):
    refs = [ref.strip() for ref in refs]
    preds = [pred.strip() for pred in preds]

    results = {}
    results["exact_match"] = exact_match.compute(predictions=preds, references=refs)["exact_match"]
    results["bleu"] = hf_bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    rouge_scores = rouge.compute(predictions=preds, references=refs)
    results["rouge1"] = rouge_scores["rouge1"]
    results["rouge2"] = rouge_scores["rouge2"]
    results["rougeL"] = rouge_scores["rougeL"]

    smoothie = SmoothingFunction().method4
    nltk_bleu = sum(
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        for pred, ref in zip(preds, refs)
    ) / len(preds)
    results["nltk_bleu"] = nltk_bleu

    return results

test_metrics = compute_all_metrics(test_preds, test_df["cmd"].tolist())

test_results_df = test_df.copy()
test_results_df["predicted_cmd"] = test_preds
test_results_df["exact_match"] = test_results_df.apply(
    lambda row: row["cmd"].strip() == row["predicted_cmd"].strip(), axis=1
)

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "gpt2_test_predictions.csv")
test_results_df[["invocation", "cmd", "predicted_cmd", "exact_match"]].to_csv(csv_path, index=False)

print("\n📊 Final Evaluation Summary (Test Set Only)\n")
print(f"{'Metric':<15}{'Test':>12}")
print("-" * 30)
for metric in ["exact_match", "bleu", "nltk_bleu", "rouge1", "rouge2", "rougeL"]:
    print(f"{metric:<15}{test_metrics[metric]:>12.3f}")

print("\n⏱️ Inference Time Summary")
print("-" * 40)
print(f"{'Test Total Time (s)':<25}{test_gen_time:.2f}")
print(f"{'Test Time per Sample (s)':<25}{test_gen_time / len(test_preds):.4f}")
print(f"\n📁 Test predictions saved to: {csv_path}")

def predict_from_input(prompt, model, tokenizer, max_new_tokens=20):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens)
    end_time = time.time()

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    predicted_cmd = decoded.split("<|sep|>")[-1].strip()
    elapsed = end_time - start_time
    return predicted_cmd, elapsed

log_path = os.path.join(output_dir, "gpt2_user_predictions_log.csv")
if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("invocation,predicted_cmd,generation_time,bleu_score\n")

def compute_bleu(pred, ref):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)

while True:
    user_input = input("\n💬 Enter a natural language instruction (or type 'exit' to quit):\n> ").strip()
    if user_input.lower() == "exit":
        print("👋 Exiting.")
        break

    predicted, time_taken = predict_from_input(user_input, gpt2_model, gpt2_tokenizer)
    print(f"🔧 Predicted command: {predicted}")
    print(f"⏱️ Generation time: {time_taken:.4f} seconds")

    ref_cmd = input("✅ (Optional) Enter actual command for BLEU score (or press enter to skip):\n> ").strip()
    bleu_score = compute_bleu(predicted, ref_cmd) if ref_cmd else ""

    if ref_cmd:
        print(f"📏 BLEU score: {bleu_score:.4f}")

    with open(log_path, "a") as f:
        f.write(f'"{user_input}","{predicted}",{time_taken:.4f},{bleu_score}\n')
