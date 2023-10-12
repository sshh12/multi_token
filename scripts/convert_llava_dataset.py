import argparse
import json
import os

from datasets import Dataset


def _convert_convo(convo):
    msgs = []
    for m in convo:
        msgs.append(
            {
                "role": {"gpt": "assistant", "human": "user"}[m["from"]],
                "content": m["value"],
            }
        )
    return msgs


def main(args):
    def gen():
        with open(args.llava_json) as f:
            data = json.load(f)
        for row in data:
            fn = os.path.join(args.image_folder, "GCC_train_" + row["image"][3:])
            if not os.path.exists(fn):
                continue
            yield {
                "id": row["id"],
                "images": [fn],
                "messages": _convert_convo(row["conversations"]),
            }

    ds = Dataset.from_generator(gen)
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava_json", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()
    main(args)
