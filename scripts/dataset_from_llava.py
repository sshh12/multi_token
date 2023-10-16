from typing import List
import argparse
import json
import os

from datasets import Dataset


def _convert_convo(convo) -> List:
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
    def gen(json_fns):
        for json_fn in json_fns:
            with open(json_fn) as f:
                data = json.load(f)
            for row in data:
                img_path = row["image"][3:]
                if "_train_" in img_path:
                    img_path = img_path.replace("_train_", "")
                fn = os.path.join(args.image_folder, "GCC_train_" + img_path)
                if not os.path.exists(fn):
                    continue
                yield {
                    "id": row["id"],
                    "images": [fn],
                    "messages": _convert_convo(row["conversations"]),
                }

    ds = Dataset.from_generator(
        gen, gen_kwargs={"json_fns": args.llava_json}, num_proc=len(args.llava_json)
    )
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--llava_json", type=str, action="append")
    parser.add_argument("-f", "--image_folder", type=str)
    parser.add_argument("-o", "--output_folder", type=str)
    args = parser.parse_args()
    main(args)
