from typing import List
import argparse
import re
import glob
import json

from datasets import load_dataset
from datasets import Dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER
from multi_token.modalities.document_gte import (
    split_text_into_documents,
)

TEMP_TOKEN = "<<<TEMP-TOKEN>>>"

# regex, doc, prompt
LONG_ALPACA_REGEXES = [
    (
        r"Below is a paper. Memorize the paper and answer my question after the paper.\n The paper begins. \n ([\s\S]+) \n Now the paper ends. \n([\s\S]+)",
        lambda m: m.group(1),
        lambda m: f"Read the paper {TEMP_TOKEN}. {m.group(2)}",
    ),
    (
        r"Below is a paper. Memorize the material and answer my question after the paper.\n([\s\S]+)\n Now the material ends. ([\s\S]+)",
        lambda m: m.group(1),
        lambda m: f"Read the paper {TEMP_TOKEN}. {m.group(2)}",
    ),
    (
        r"There are two papers. Memorize them and answer my question after the paper.\n The first paper begins. \n ([\s\S]+) Now the second paper ends.([\s\S]+)",
        lambda m: m.group(1),
        lambda m: f"Read the papers {TEMP_TOKEN}. {m.group(2)}",
    ),
    (
        r"Below is some paragraphs in the book, ([\s\S]+?). Memorize the content and answer my question after the book.\n([\s\S]+) \n Now the material ends.([\s\S]+)",
        lambda m: m.group(2),
        lambda m: f"Read the book {m.group(1)} {TEMP_TOKEN}. {m.group(3)}",
    ),
]

# regex, doc, prompt, answer
LONG_DATA_REGEXES = [
    (
        r"Write a high-quality answer for the given question using only the provided search results \(some of which might be irrelevant\).([\s\S]+)Question: ([\s\S]+)Answer: ([\s\S]+)\nLong Answer: ([\s\S]+)",
        lambda m: m.group(1).strip(),
        lambda m: f"Write a high-quality answer for the given question using only the provided search results {TEMP_TOKEN}. {m.group(2).strip()}",
        lambda m: m.group(4).strip(),
    ),
    (
        r"([\s\S]+)\nQ: ([\s\S]+)\nA: ([\s\S]+)",
        lambda m: m.group(1).strip(),
        lambda m: f"Read the following book {TEMP_TOKEN}. {m.group(2).strip()}",
        lambda m: m.group(3).strip(),
    ),
]


def _write_long_alpaca_convo(row, max_document_chunks) -> List:
    doc_text = None
    prompt = None
    for regex, get_doc, get_prompt in LONG_ALPACA_REGEXES:
        match = re.match(regex, row["instruction"])
        if match:
            doc_text = get_doc(match)
            prompt = get_prompt(match).replace("Question: ", "")
            break

    if doc_text is None and row["input"]:
        doc_text = row["input"]
        prompt = row["instruction"] + f" {TEMP_TOKEN}"

    if doc_text is None:
        raise ValueError("No document found")

    docs = split_text_into_documents(doc_text)
    if len(docs) > max_document_chunks:
        raise ValueError("Document too long")
    example = {
        "id": "longalpaca-" + str(hash(row["instruction"])),
        "documents": docs,
    }
    example["messages"] = [
        {
            "role": ROLE_USER,
            "content": prompt.replace(TEMP_TOKEN, "<document>" * len(docs)),
        },
        {
            "role": ROLE_ASSISTANT,
            "content": row["output"].replace("Answer: ", ""),
        },
    ]
    return example


def _write_long_data_collections_convo(row, max_document_chunks) -> List:
    doc_text = None
    prompt = None
    answer = None
    for regex, get_doc, get_prompt, get_answer in LONG_DATA_REGEXES:
        match = re.match(regex, row["text"])
        if match:
            doc_text = get_doc(match)
            prompt = get_prompt(match)
            answer = get_answer(match).replace(" .", ".")
            break

    if not doc_text or not prompt or not answer:
        raise ValueError("No document found")

    docs = split_text_into_documents(doc_text)
    if len(docs) > max_document_chunks:
        raise ValueError("Document too long")
    example = {
        "id": "longdatacollection-" + str(hash(row["text"])),
        "documents": docs,
    }
    example["messages"] = [
        {
            "role": ROLE_USER,
            "content": prompt.replace(TEMP_TOKEN, "<document>" * len(docs)),
        },
        {
            "role": ROLE_ASSISTANT,
            "content": answer,
        },
    ]
    return example


def main(args):
    long_alpaca = load_dataset(args.long_alpaca_path, "train")["train"]

    def gen():
        for row in long_alpaca:
            try:
                yield _write_long_alpaca_convo(row, args.max_document_chunks)
            except ValueError:
                continue
        for long_collection_fn in glob.iglob(args.long_collections_glob):
            with open(long_collection_fn) as f:
                for line in f:
                    row = json.loads(line)
                    try:
                        yield _write_long_data_collections_convo(
                            row, args.max_document_chunks
                        )
                    except ValueError:
                        continue

    ds = Dataset.from_generator(gen)
    ds = ds.shuffle(seed=42)
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--long_alpaca_path", type=str, default="Yukang/LongAlpaca-12k")
    parser.add_argument("--long_collections_glob", type=str)
    parser.add_argument("-o", "--output_folder", type=str)
    parser.add_argument("-c", "--max_document_chunks", type=int, default=256)
    args = parser.parse_args()
    main(args)
