from typing import List
import random
import argparse
import json

from huggingface_hub import hf_hub_download
from datasets import Dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER

PRETRAIN_PHRASES = [
    "Repeat the content of the video <video>",
    "What is occuring in the video? <video>",
    "<video>. What happened?",
    "Convert <video> to text",
    "What is being depicted in <video>?",
    "What is the content of <video>?",
    "Describe what occurs in the video. <video>",
    "What is the video about? <video>",
    "<video>. Tell me what occurs in the video.",
    "What is the video about? <video>",
    "Give me a summary of <video>",
    "<video>. Detail what is happening in the video.",
    "Tell me about <video>",
]


def _timestamp_to_seconds(timestamp: str):
    parts = timestamp.split(":")
    seconds = float(parts[-1])
    seconds += float(parts[-2]) * 60
    seconds += float(parts[-3]) * 60 * 60
    return seconds


def _write_convo(row) -> List:
    video = {
        "url": "https://www.youtube.com/watch?v=" + row["YoutubeID"],
        "start_time": _timestamp_to_seconds(row["Start_timestamp"]),
        "end_time": _timestamp_to_seconds(row["End_timestamp"]),
    }
    # test load, jk let it fail
    # load_video(video)
    example = {
        "videos": [video],
    }
    phrase = random.choice(PRETRAIN_PHRASES)
    example["messages"] = [
        {
            "role": ROLE_USER,
            "content": phrase,
        },
        {
            "role": ROLE_ASSISTANT,
            "content": row["Caption"],
        },
    ]
    return example


def main(args):
    path = hf_hub_download(
        repo_id="OpenGVLab/InternVid", filename="caption.jsonl", repo_type="dataset"
    )

    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    print("Dataset size:", len(rows))

    if len(rows) > args.max_examples:
        rows = random.sample(rows, k=args.max_examples)

    def gen(subset_rows):
        for row in subset_rows:
            try:
                yield _write_convo(row)
            except Exception as e:
                print(e)

    ds = Dataset.from_generator(gen, gen_kwargs={"subset_rows": rows}, num_proc=5)
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_folder", type=str, default="/data/xclip-internvid-pretrain"
    )
    parser.add_argument("-n", "--max_examples", type=int, default=500_000)
    args = parser.parse_args()
    main(args)
