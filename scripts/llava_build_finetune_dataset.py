from typing import List
import argparse
import json
import os

from datasets import Dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER


def _convert_convo(convo) -> List:
    msgs = []
    for m in convo:
        msgs.append(
            {
                "role": {"gpt": ROLE_ASSISTANT, "human": ROLE_USER}[m["from"]],
                "content": m["value"],
            }
        )
    return msgs


def _fix_path(path):
    parts = path.split("/")
    parts = [parts[0], parts[1], parts[1], *parts[2:]]
    new_path = os.path.join(*parts)
    return new_path


def main(args):
    rows = []
    for json_fn in args.llava_json:
        with open(json_fn) as f:
            rows.extend(json.load(f))

    def gen(rows):
        for row in rows:
            try:
                img_path = row["image"]
            except KeyError:
                continue
            fn = os.path.join(args.image_folder, _fix_path(img_path))
            if not os.path.exists(fn):
                print("Skipping", fn)
                continue
            yield {
                "id": str(row["id"]),
                "images": [fn],
                "messages": _convert_convo(row["conversations"]),
            }

    ds = Dataset.from_generator(gen, gen_kwargs={"rows": rows}, num_proc=args.num_proc)
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--llava_json", type=str, action="append")
    parser.add_argument("-f", "--image_folder", type=str)
    parser.add_argument("-o", "--output_folder", type=str)
    parser.add_argument("-n", "--num_proc", type=int, default=1)
    args = parser.parse_args()
    main(args)
