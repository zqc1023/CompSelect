from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import argparse


import random

all_time = []


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


def get_extract_sentences(query, docs, tokenizer, model):
    # best prompt for extract all useful sentences
    final_docs = "\n".join([f"document {j}: {doc}" for j, doc in enumerate(docs, 1)])
    system_prompt = """You are a highly skilled assistant specializing in extracting relevant information from provided documents. Your task is to identify and extract sentences from the documents as much as possible that are most directly useful for answering the given question. Rank the sentences in order of relevance, with the most relevant sentence listed first. Preface each sentence with its sequence number as follows:
    Sentence 1: 
    ......
    Sentence n: 
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\nDocuments: {final_docs}"},
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

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.01,
            pad_token_id=pad_token_id,
            eos_token_id=terminators,
        )
    end_event.record()
    torch.cuda.synchronize()
    gpu_time = start_event.elapsed_time(end_event)

    all_time.append(gpu_time)

    response = outputs[0][input_ids.shape[-1] :]

    return tokenizer.decode(response, skip_special_tokens=True)


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


def main():
    args = parse_args()

    question, answer, documents = load_qa_documents(args.input)
    model, tokenizer = load_llama(args.model)

    for i in tqdm(range(len(question))):
        try:

            summary = get_extract_sentences(question[i], documents[i], tokenizer, model)
            with open(args.output, "a+", encoding="utf-8") as f:
                res = {
                    "question": question[i],
                    "answer": answer[i],
                    "documents": documents[i],
                    "summary": summary,
                }
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")

        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
