import json
import torch
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import *


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name, torch_dtype, device_map):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def clean_and_split_sentences(summary_list):
    if isinstance(summary_list, list):
        return summary_list
    return [s.strip() for s in summary_list.split(".") if s.strip()]


def load_qa_documents(data_path):
    q, a, docs, summary = [], [], [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                q.append(d["question"])
                a.append(d["answer"])
                docs.append(d["documents"])
                summary.append(clean_and_split_sentences(d["summary"]))
            except Exception as e:
                print(f"Error: {e}")
    return q, a, docs, summary


def get_answer(
    documents, question, model, tokenizer, max_length, max_new_tokens, temperature
):
    final_docs = "\n".join(
        [f"Sentence {j}: {doc}" for j, doc in enumerate(documents, 1)]
    )

    system_prompt = (
        "You are a helpful, respectful, and honest assistant. "
        "Answer the question with couple of words using the provided documents. "
        "For example: Question: What is the capital of France? Output: Paris."
    )

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
        max_length=max_length,
    )

    input_ids = tokenized.to(model.device)
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=terminators,
        )

    new_tokens = output_ids[0, input_ids.shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


def adaptive_truncate(
    q_list,
    a_list,
    docs_list,
    summary_list,
    model,
    tokenizer,
    save_path,
    max_length,
    max_new_tokens,
    temperature,
):
    results = []

    for i in tqdm(range(len(q_list)), desc="Processing"):
        question = q_list[i]
        answer = a_list[i]
        documents = docs_list[i]
        summary = summary_list[i]

        final_subset = []
        found = False
        for k in range(len(summary), 0, -1):
            subset = summary[:k]
            ans = get_answer(
                subset,
                question=question,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            score = sub_exact_match(ans, answer)

            if score == 1.0:
                final_subset = subset
                found = True
                break

        if not found:
            final_subset = []

        results.append(
            {
                "question": question,
                "answer": answer,
                "documents": documents,
                "summary": summary,
                "final": final_subset,
            }
        )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive truncation script")

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save output JSON"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name or path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="Torch dtype (e.g., float16, bfloat16)",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="Device map setting"
    )
    parser.add_argument(
        "--max_length", type=int, default=3096, help="Max token length for input"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Sampling temperature"
    )

    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_name, dtype, args.device_map)
    q, a, docs, summary = load_qa_documents(args.data_path)

    adaptive_truncate(
        q,
        a,
        docs,
        summary,
        model,
        tokenizer,
        args.save_path,
        args.max_length,
        args.max_new_tokens,
        args.temperature,
    )


if __name__ == "__main__":
    main()
