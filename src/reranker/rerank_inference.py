import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import argparse

import random


def set_seed(seed=42):

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    print(f"Model loaded successfully on {device}")
    return model


def rank_documents(model, query, documents):

    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    cosine_similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    sorted_indices = cosine_similarities.argsort(descending=True)

    sorted_results = [documents[idx] for idx in sorted_indices]

    return sorted_results


def clean_and_split_sentences(input_string):

    sentences = input_string.split("\n")
    cleaned_sentences = [
        sentence.split(": ", 1)[1] if ": " in sentence else sentence
        for sentence in sentences
    ]
    return cleaned_sentences


def load_qa_documents(file_path):

    questions, answers, documents, summaries = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                questions.append(data["question"])
                answers.append(data["answer"])
                documents.append(data["documents"])
                summaries.append(clean_and_split_sentences(data["summary"]))
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Error: {e}")
    return questions, answers, documents, summaries


def save_ranked_results(output_path, question, answer, documents, ranked_summary):

    formatted_summary = "\n".join(
        [f"Sentence {i+1}: {s}" for i, s in enumerate(ranked_summary)]
    )
    res = {
        "question": question,
        "answer": answer,
        "documents": documents,
        "summary": formatted_summary,
    }
    with open(output_path, "a+", encoding="utf-8") as f:
        json.dump(res, f)
        f.write("\n")


def main(args):
    model = load_model(args.model_path)
    questions, answers, documents, summaries = load_qa_documents(args.input_file)

    for i in tqdm(range(len(questions))):
        ranked_summary = rank_documents(model, questions[i], summaries[i])
        save_ranked_results(
            args.output_file, questions[i], answers[i], documents[i], ranked_summary
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank QA summaries using SentenceTransformer model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained reranker model"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input QA JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save ranked results"
    )
    args = parser.parse_args()
    main(args)
