import json
import os

path = "/path/to/your/input_file.json"  # specify your input file path here
data = [json.loads(line.strip()) for line in open(path, "r")]

final_data = []


system_prompt = """You are a highly skilled assistant specializing in extracting relevant information from provided documents. Your task is to identify and extract sentences from the documents as much as possible that are most directly useful for answering the given question. Rank the sentences in order of relevance, with the most relevant sentence listed first. Preface each sentence with its sequence number as follows:
Sentence 1: 
......
Sentence n: 
"""

for item in data:
    docs = "\n".join(
        [f"document {j}: {doc}" for j, doc in enumerate(item["documents"], start=1)]
    )
    sentences = "\n".join(
        [
            f"Sentence {j}: {sentence}"
            for j, sentence in enumerate(item["sentences"], start=1)
        ]
    )

    final_data.append(
        {
            "instruction": f"Question: {item['question']}\nDocuments: {docs}",
            "input": "",
            "output": sentences,
            "system": system_prompt,
            "history": [],
        }
    )
print(len(final_data))
save_path = f"{path}_for_finetuned.json"
with open(
    save_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(final_data, f, indent=4)
