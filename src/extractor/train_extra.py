import random
import os
import json
import torch
import argparse
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import logging
import sys


IGNORE_INDEX = -100


class LoggerWriter:
    """Redirect stdout/stderr to logging, flush immediately."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)
            for handler in self.logger.handlers:
                handler.flush()

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaMA3 Extractor with LoRA")
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--gpu", type=str, default="1", help="gpu_id")
    parser.add_argument("--model", type=str, help="model_name")
    return parser.parse_args()


def create_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} Created")
    else:
        print(f"Folder {folder_path} Existed")


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_gpu(gpu_id: str):
    if gpu_id.lower() == "all":
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def setup_logger(output_dir):
    log_file = os.path.join(output_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. Logs will be saved to {log_file}")

    sys.stdout = LoggerWriter(logger, logging.INFO)

    return logger


def load_json(file_path, max_samples=20000):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data[:max_samples]


def build_prompt(question, documents, sentences):
    system_prompt = """You are a highly skilled assistant specializing in extracting relevant information from provided documents. Your task is to identify and extract sentences from the documents as much as possible that are most directly useful for answering the given question. Rank the sentences in order of relevance, with the most relevant sentence listed first. Preface each sentence with its sequence number as follows:
    Sentence 1: 
    ......
    Sentence n: 
    """
    docs_str = "\n".join(
        [f"document {j}: {doc}" for j, doc in enumerate(documents, start=1)]
    )
    user_prompt = f"Question: {question}\nDocuments: {docs_str}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": sentences},
    ]
    return messages


def format_dataset(raw_data, tokenizer, max_length=2048):
    formatted = []
    for d in raw_data:
        question = d["question"]
        documents = d["documents"]

        target = "\n".join(
            [
                f"Sentence {j}: {sentence}"
                for j, sentence in enumerate(d["sentences"], start=1)
            ]
        )

        messages = build_prompt(question, documents, sentences=target)

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def train_extractor():
    args = parse_args()
    set_seed(42)
    set_gpu(args.gpu)

    dataset_name = os.path.basename(os.path.dirname(args.input))
    knn_param = os.path.splitext(os.path.basename(args.input))[0].split("_")[-1]
    model_name = os.path.basename(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./saves/{dataset_name}/{model_name}/knn_{knn_param}_{timestamp}"

    create_path(output_dir)
    create_path(os.path.join(output_dir, "logs"))
    setup_logger(output_dir)

    raw_data = load_json(args.input)
    random.shuffle(raw_data)

    split_idx = int(0.85 * len(raw_data))
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]

    logging.info(f"Loaded {len(train_data)} train samples from {args.input}")
    logging.info(f"Loaded {len(val_data)} eval samples from {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    def preprocess_function(examples):

        tokenized = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=2048
        )

        labels = tokenized["input_ids"].copy()
        text = examples["text"]

        assistant_start = text.rfind("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:

            prefix_ids = tokenizer(
                text[
                    : assistant_start
                    + len("<|start_header_id|>assistant<|end_header_id|>")
                ],
                truncation=True,
                max_length=2048,
            )["input_ids"]
            cutoff = len(prefix_ids)
            labels[:cutoff] = [IGNORE_INDEX] * cutoff
        else:

            labels = [IGNORE_INDEX] * len(labels)

        tokenized["labels"] = labels
        return tokenized

    peft_config = LoraConfig(
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    train_dataset = format_dataset(train_data, tokenizer)
    logging.info(train_dataset[0])
    eval_dataset = format_dataset(val_data, tokenizer)
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=False)
    tokenized_val_dataset = eval_dataset.map(preprocess_function, batched=False)
    logging.info(tokenized_train_dataset[0])

    sft_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        num_train_epochs=16,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=800,
        bf16=True,
        fp16=False,
        save_total_limit=8,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=sft_args,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info((f"Model saved to {output_dir}"))


if __name__ == "__main__":
    train_extractor()
