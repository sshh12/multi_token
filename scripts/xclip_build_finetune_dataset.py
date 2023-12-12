from typing import List
import random
import argparse
import json

from datasets import Dataset, load_dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER


def _write_convo(row) -> List:
    video = "https://www.youtube.com/watch?v=" + row["video_id"][2:]
    # test load, jk let it fail
    # load_video(video)
    example = {
        "videos": [video],
    }
    example["messages"] = [
        {
            "role": ROLE_USER,
            "content": row["q"],
        },
        {
            "role": ROLE_ASSISTANT,
            "content": row["a"],
        },
    ]
    return example


def main(args):
    data = load_dataset("MBZUAI/VideoInstruct-100K", split="train")

    def gen():
        for row in data:
            try:
                yield _write_convo(row)
            except Exception as e:
                print(e)

    ds = Dataset.from_generator(gen)
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_folder", type=str, default="/data/xclip-videoinstruct-finetune"
    )
    args = parser.parse_args()
    main(args)
