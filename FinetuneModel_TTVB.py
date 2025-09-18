import os
import random
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# ========== User config ==========
ROOT_DIR = "C:/Python/archive/clusters/clusters"  
USE_REF = "ref1"                   
MODEL_NAME = "VietAI/vit5-base"       
OUTPUT_DIR = "./results_cluster_finetune"
MAX_INPUT_TOKENS = 1024
MAX_TARGET_TOKENS = 128
SEED = 42
TRAIN_FRAC = 0.9                  
CACHE_DIR = "./cache_tokenized"
# =================================

random.seed(SEED)

# --- Load model/tokenizer globally for modularity ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def read_text_if_exists(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

def build_examples_from_cluster(cluster_path: Path):
    cluster_id = cluster_path.name
    doc_bodies = []
    for p in sorted(cluster_path.glob("*.body.txt")):
        body = read_text_if_exists(p)
        if body:
            doc_bodies.append(body)
    if len(doc_bodies) == 0:
        for p in sorted(cluster_path.glob("*.txt")):
            name = p.name
            if name.startswith(cluster_id):
                continue
            content = read_text_if_exists(p)
            if content:
                doc_bodies.append(content)
    if len(doc_bodies) == 0:
        return None

    text = "\n\n".join(doc_bodies)

    ref1 = read_text_if_exists(cluster_path / f"{cluster_id}.ref1.txt")
    ref2 = read_text_if_exists(cluster_path / f"{cluster_id}.ref2.txt")
    if USE_REF == "ref1":
        summary = ref1 if ref1 else (ref2 if ref2 else "")
    elif USE_REF == "ref2":
        summary = ref2 if ref2 else (ref1 if ref1 else "")
    elif USE_REF == "merge":
        parts = []
        if ref1: parts.append(ref1)
        if ref2: parts.append(ref2)
        summary = "\n".join(parts)
    else:
        summary = ref1 or ref2 or ""

    info_text = read_text_if_exists(cluster_path / f"{cluster_id}.info")
    meta = {"cluster_info": info_text} if info_text else {}

    return {"cluster_id": cluster_id, "text": text, "summary": summary, "meta": meta}

def load_all_clusters(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root_dir} không tồn tại.")
    clusters = []
    for cluster_dir in sorted(root.iterdir()):
        if cluster_dir.is_dir():
            ex = build_examples_from_cluster(cluster_dir)
            if ex and ex["summary"].strip() != "":
                clusters.append(ex)
    return clusters

def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["summary"]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_TOKENS,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def get_tokenized_datasets(train_ds, val_ds):
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(exist_ok=True)
    cache_file = cache_path / "tokenized.arrow"
    if cache_path.exists() and any(cache_path.iterdir()):
        print("Loading tokenized dataset from cache...")
        return DatasetDict.load_from_disk(str(cache_path))
    else:
        print("Tokenizing datasets...")
        tokenized_train = train_ds.map(
            preprocess_function, batched=True, remove_columns=train_ds.column_names
        )
        tokenized_val = val_ds.map(
            preprocess_function, batched=True, remove_columns=val_ds.column_names
        )
        tokenized = DatasetDict({
            "train": tokenized_train,
            "validation": tokenized_val
        })
        tokenized.save_to_disk(str(cache_path))
        return tokenized

def main():
    clusters = load_all_clusters(ROOT_DIR)
    if len(clusters) == 0:
        raise SystemExit("Không tìm thấy cụm hợp lệ trong ROOT_DIR.")

    random.shuffle(clusters)
    n_train = int(len(clusters) * TRAIN_FRAC)
    train_clusters = clusters[:n_train]
    val_clusters = clusters[n_train:]

    train_df = pd.DataFrame(train_clusters)
    val_df = pd.DataFrame(val_clusters)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tokenized_datasets = get_tokenized_datasets(train_ds, val_ds)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        return preds, labels

    def compute_metrics(eval_pred):
        preds_ids, labels_ids = eval_pred
        decoded_preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return {k: round(v * 100, 2) for k, v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        eval_strategy="epoch",
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=50,
        save_total_limit=3,
        fp16=False,
        remove_unused_columns=True,
        push_to_hub=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tune xong. Lưu kết quả tại:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
