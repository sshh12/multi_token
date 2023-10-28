from typing import List
import random
import argparse

from datasets import load_dataset
from datasets import Dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER
from multi_token.modalities.document_gte import (
    split_text_into_documents,
)

TEMP_TOKEN = "<<<TEMP-TOKEN>>>"

PRETRAIN_PHRASES = [
    f"Repeat the content of the document {TEMP_TOKEN}",
    f"Transcribe {TEMP_TOKEN}",
    f"Provide a verbatim transcription of {TEMP_TOKEN}",
    f"Write down exactly what is in {TEMP_TOKEN}",
    f"Copy the text from {TEMP_TOKEN}",
    f"Duplicate the content of {TEMP_TOKEN}",
    f"Reproduce the text in {TEMP_TOKEN}",
    f"Render the exact text from {TEMP_TOKEN}",
    f"Echo the content of {TEMP_TOKEN}",
    f"Mirror the text in {TEMP_TOKEN}",
    f"Reflect the content of {TEMP_TOKEN}",
    f"Transcribe the exact words from {TEMP_TOKEN}",
    f"Write out the exact content of {TEMP_TOKEN}",
    f"Provide a direct transcription of {TEMP_TOKEN}",
    f"Give a word-for-word account of {TEMP_TOKEN}",
    f"Reiterate the exact text of {TEMP_TOKEN}",
    f"Replicate the content of {TEMP_TOKEN}",
    f"Reprint the text from {TEMP_TOKEN}",
    f"Rewrite the exact words from {TEMP_TOKEN}",
]


def _write_convo(row, max_document_chunks) -> List:
    docs = split_text_into_documents(row["text"])
    if len(docs) > max_document_chunks:
        raise ValueError("Document too long")
    example = {
        "id": str(row["title"]),
        "documents": docs,
    }
    phrase = random.choice(PRETRAIN_PHRASES)
    example["messages"] = [
        {
            "role": ROLE_USER,
            "content": phrase.replace(TEMP_TOKEN, "<document>" * len(docs)),
        },
        {
            "role": ROLE_ASSISTANT,
            "content": row["text"],
        },
    ]
    return example


def main(args):
    wiki_data = load_dataset("graelo/wikipedia", "20230601.en")["train"]

    idxs = list(range(len(wiki_data)))
    random.shuffle(idxs)

    def gen():
        i = 0
        for idx in idxs:
            row = wiki_data[idx]
            try:
                yield _write_convo(row, args.max_document_chunks)
            except ValueError:
                pass
            else:
                i += 1
                if i >= args.max_examples:
                    break

    ds = Dataset.from_generator(gen)
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder", type=str)
    parser.add_argument("-n", "--max_examples", type=int, default=1_000_000)
    parser.add_argument("-c", "--max_document_chunks", type=int, default=4)
    args = parser.parse_args()
    main(args)
