import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import argparse
from metrics import *
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA with LLaMA model and permuted docs"
    )
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--model", type=str, default="llama3", help="model_name")
    parser.add_argument("--top", type=int, required=True, help="top_k")
    parser.add_argument(
        "--mode", type=int, default=0, help="answer_containing_sentences"
    )

    return parser.parse_args()


def load_llama(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def load_qa_documents(data_path):
    q, a, docs = [], [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                q.append(d["question"])
                a.append(d["answer"])
                docs.append(d["documents"])
            except Exception as e:
                print(f"Error: {e}")
    return q, a, docs


print_count = 0


def get_answer(documents, question, model, tokenizer, top_k, mode, answers=None):
    global print_count

    if mode == 1:
        all_sentences = []
        for doc in documents[0:top_k]:
            sentences = re.split(r"(?<=[.!?]) +", doc)
            all_sentences.extend(sentences)

        selected_sentences = set()
        for ans in answers:
            for sent in all_sentences:
                if ans in sent:
                    selected_sentences.add(sent)

        final_docs = "\n".join(
            [
                f"document {j}: {sent}"
                for j, sent in enumerate(list(selected_sentences), 1)
            ]
        )

    else:
        final_docs = "\n".join(
            [f"document {j}: {doc}" for j, doc in enumerate(documents[0:top_k], 1)]
        )

    system_prompt = "You are a helpful, respectful, and honest assistant. Answer the question with couple of words using the provided documents. For example: Question: What is the capital of France? Output: Paris."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nDocuments: {final_docs}"},
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    )

    input_ids = tokenized.to(model.device)

    if print_count < 5:
        print(f"[Sample {print_count+1}] Token length: {input_ids.shape[-1]}")
        print_count += 1

    if isinstance(input_ids, dict):
        attention_mask = input_ids["attention_mask"]
        input_ids = input_ids["input_ids"]
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.01,
            pad_token_id=pad_token_id,
            eos_token_id=terminators,
        )

    new_tokens = output_ids[0, input_ids.shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer


def main():
    args = parse_args()
    question, answer, documents = load_qa_documents(args.input)
    model, tokenizer = load_llama(args.model)
    number = 0
    answer_set = []
    total_time = 0

    for i in tqdm(range(len(question))):
        ans = get_answer(
            documents=documents[i],
            question=question[i],
            model=model,
            tokenizer=tokenizer,
            top_k=args.top,
            mode=args.mode,
            answers=answer[i] if args.mode == 1 else None,
        )

        answer_set.append(ans.strip().rstrip("."))

        number += sub_exact_match(ans, answer[i])

        avg_f1, _ = dataset_level_f1(answer_set, answer[0 : len(answer_set)])

    avg_score, _ = batch_sub_exact_match(answer_set, answer)
    print("avgEM:", avg_score)
    avg_f1, _ = dataset_level_f1(answer_set, answer)
    print("avgF1:", avg_f1)


if __name__ == "__main__":
    main()
