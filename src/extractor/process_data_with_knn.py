import json
import re
import torch
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from nltk.tokenize import sent_tokenize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--output", type=str, required=True, help="output_file")
    parser.add_argument(
        "--threshold", type=float, default=0.25, help="similarity threshold"
    )
    return parser.parse_args()


def split_into_sentences(paragraph):
    final_sentencs = sent_tokenize(paragraph)

    return final_sentencs


def finding_sentence(answer, sentences):
    result = set()
    for ans in answer:
        for sentence in sentences:
            if ans in sentence:
                result.add(sentence)
    return list(result)


def filter_sentences_by_similarity(model, a, b, threshold):

    a_embeddings = model.encode(a, convert_to_tensor=True)
    b_embeddings = model.encode(b, convert_to_tensor=True)

    distances = cosine_distances(a_embeddings.cpu().numpy(), b_embeddings.cpu().numpy())

    neighbors = set()
    for i, _ in enumerate(a):
        for j, b_sentence in enumerate(b):
            if distances[i, j] <= threshold:
                neighbors.add(b_sentence)
    return list(neighbors)


def sorted_sentences(model, query, sentences):
    e_q = model.encode([query], convert_to_tensor=True)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarities = cosine_similarity(e_q.cpu().numpy(), embeddings.cpu().numpy())[0]
    sorted_sentences_list = [
        s
        for s, _ in sorted(
            zip(sentences, similarities), key=lambda x: x[1], reverse=True
        )
    ]
    return sorted_sentences_list


def main():

    args = parse_args()

    model_path = "/path/to/your/sentence-transformers-model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_path)
    if torch.cuda.is_available():
        model = model.to(device)

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    all_number = 0
    number = 0

    for item in data:
        answer = item["answer"]
        documents = item["documents"]
        sentences = []
        for doc in documents:
            sentences.extend(split_into_sentences(doc))

        result = finding_sentence(answer, sentences)
        if not result:
            continue
        all_number += 1
        if args.threshold != 0.0:

            filtered_results = filter_sentences_by_similarity(
                model, result, sentences, threshold=args.threshold
            )
            if len(result) != len(filtered_results):
                number += 1

        else:
            filtered_results = result

        final_filtered_result = sorted_sentences(
            model, item["question"], filtered_results
        )

        with open(args.output, "a+", encoding="utf-8") as f_out:
            res = {
                "question": item["question"],
                "answer": item["answer"],
                "documents": item["documents"],
                "sentences": final_filtered_result,
            }
            json.dump(res, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()
