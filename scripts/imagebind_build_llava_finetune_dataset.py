from typing import List
import argparse
import random
import json
import os

from datasets import Dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER


TYPES = ["audio", "image", "text"]

REPLACEMENTS = {
    "image": ["audio", "image", "document"],
    "picture": ["audio file", "picture", "text snippet"],
    "photo": ["sound", "photo", "text"],
    "visual": ["audio", "visual", "textual"],
    "see": ["hear", "see", "read"],
    "look": ["sound", "look", "read"],
    "visible": ["audible", "visible", "readable"],
}

TEMP_TOKEN = "<<<TEMP-TOKEN>>>"

EXCLUDE_WORDS = ["region", "ocr", "color", "right", "left"]


def _convert_convo(convo) -> List:
    type_idx = TYPES.index(random.choice(TYPES))
    msgs = []
    for m in convo:
        content = m["value"].replace("<image>", TEMP_TOKEN)
        for k, v in REPLACEMENTS.items():
            content = content.replace(k, v[type_idx])
        content = content.replace(TEMP_TOKEN, "<imagebind>")
        msgs.append(
            {
                "role": {"gpt": ROLE_ASSISTANT, "human": ROLE_USER}[m["from"]],
                "content": content,
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

            # avoid tasks too image-y
            convo_text = repr(row["conversations"]).lower()

            if "ocr" in img_path or any(w in convo_text for w in EXCLUDE_WORDS):
                continue

            fn = os.path.join(args.image_folder, _fix_path(img_path))
            if not os.path.exists(fn):
                print("Skipping (does not exist)", fn)
                continue
            yield {
                "id": str(row["id"]),
                "imagebinds": [fn],
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
