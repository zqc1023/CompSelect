# file: reranker_trainer.py
import sys
import os
import json
import logging
from datetime import datetime
import random
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from sentence_transformers import (
    SentenceTransformer,
    models,
    evaluation,
    losses,
    InputExample,
)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
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

    sys.stdout = LoggerWriter(logger, logging.INFO)

    logger.info(f"Logger initialized. Logs will be saved to {log_file}")
    return logger


class RERANKdataset:
    def __init__(self, json_file_path):
        self.data_df = []
        with open(json_file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                question = data.get("question", "")
                positive_texts = data.get("positive", [])
                negative_texts = data.get("negative", [])

                if positive_texts and negative_texts:
                    for pos_text in positive_texts:
                        for neg_text in negative_texts:
                            self.data_df.append((question, pos_text, neg_text))
        logging.info(f"Loaded {len(self.data_df)} total triplets from {json_file_path}")

    def sample_and_split(self, max_samples=None, dev_ratio=0.15):
        data = self.data_df
        if max_samples:
            data = data[:max_samples]
            logging.info(f"Selected {len(data)} samples from dataset")
        train_data, dev_data = train_test_split(
            data, test_size=dev_ratio, random_state=42
        )
        logging.info(
            f"Split into {len(train_data)} train and {len(dev_data)} dev samples"
        )
        return train_data, dev_data


def train_reranker(
    data_path,
    model_name,
    output_dir,
    train_batch_size,
    max_seq_length,
    pooling,
    epochs,
    warmup_steps,
    lr,
    checkpoint_save_total_limit,
    eval_steps,
    max_train_samples=None,
):
    set_seed(42)

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print(output_dir)

    model_save_path = os.path.join(
        output_dir,
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(model_save_path, exist_ok=True)
    logger = setup_logger(model_save_path)
    logger.info(f"Model will be saved to {model_save_path}")

    dataset = RERANKdataset(data_path)
    train_data, dev_data = dataset.sample_and_split(max_samples=max_train_samples)

    train_examples = [InputExample(texts=[q, pos, neg]) for q, pos, neg in train_data]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=train_batch_size
    )

    # Loss
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Evaluator
    eval_samples = [
        {"query": q, "positive": [pos], "negative": [neg]} for q, pos, neg in dev_data
    ]
    evaluator = evaluation.RerankingEvaluator(eval_samples)

    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        checkpoint_path=model_save_path,
        evaluation_steps=eval_steps,
        checkpoint_save_total_limit=checkpoint_save_total_limit,
        optimizer_params={"lr": lr},
        save_best_model=True,
    )

    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", default="./models/distilbert-base-uncased")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--checkpoint_save_total_limit", default=3, type=int)
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--max_train_samples", default=None, type=int)
    args = parser.parse_args()

    train_reranker(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size,
        max_seq_length=args.max_seq_length,
        pooling=args.pooling,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
        eval_steps=args.eval_steps,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()
