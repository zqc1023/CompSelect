import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import random
import argparse
import random
from metrics import *


def set_seed(seed=42):

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA with LLaMA model and permuted docs"
    )
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--output", type=str, required=True, help="output_file")
    parser.add_argument("--model", type=str, default="llama3", help="model_name")

    return parser.parse_args()


def clean_and_split_sentences(input_string):
    sentences = input_string.split("\n")
    cleaned_sentences = [
        sentence.split(": ", 1)[1] if ": " in sentence else sentence
        for sentence in sentences
    ]

    return cleaned_sentences


def load_model(model_name):

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


def clean_and_split_sentences(input_string):
    sentences = input_string.split("\n")
    cleaned_sentences = [
        sentence.split(": ", 1)[1] if ": " in sentence else sentence
        for sentence in sentences
    ]
    return cleaned_sentences


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


def get_answer(documents, question, model, tokenizer):

    final_docs = "\n".join(
        [f"Sentence {j}: {doc}" for j, doc in enumerate(documents, 1)]
    )
    print(final_docs)
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
        max_length=3096,
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
            max_new_tokens=32,
            do_sample=True,
            temperature=0.01,
            pad_token_id=pad_token_id,
            eos_token_id=terminators,
        )

    new_tokens = output_ids[0, input_ids.shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(answer)
    return answer


def main():
    args = parse_args()
    question, answer, docs, sentences = load_qa_documents(args.input)
    model, tokenizer = load_model(args.model)
    for i in tqdm(range(len(question))):
        positive = []
        negative = []
        positive_id = []
        negative_id = []
        for sentence in sentences[i]:
            ans = get_answer(
                [sentence], question=question[i], model=model, tokenizer=tokenizer
            )
            if sub_exact_match(ans, answer[i]) == 1.0:
                positive.append(sentence)
                positive_id.append(sentences[i].index(sentence))
            else:
                negative.append(sentence)
                negative_id.append(sentences[i].index(sentence))

        with open(
            args.output,
            "a+",
            encoding="utf-8",
        ) as f:
            res = {
                "question": question[i],
                "answer": answer[i],
                "documents": docs[i],
                "summary": sentences[i],
                "positive": positive,
                "positive_id": positive_id,
                "negative": negative,
                "negative_id": negative_id,
            }
            json.dump(res, f)
            f.write("\n")


if __name__ == "__main__":
    main()
