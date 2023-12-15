from typing import List
import argparse
import json
import os
import random
import openai

from datasets import Dataset, load_dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER

PROMPT = """
You are helping write captions for audio clips.

Here are the tags for the audio clip you are captioning:
{captions}

Write a brief caption for the audio clip.
"""

PRETRAIN_PHRASES = [
    "What is happening in <sound>?",
    "Describe the sound. <sound>",
    "<sound> Provide a description of the audio.",
    "Can you interpret <sound>?",
    "Please explain what's happening in <sound>",
    "What does <sound> represent?",
    "Could you describe <sound> for me?",
    "What's the content of <sound>?",
    "Can you depict <sound>?",
    "What is <sound>?",
    "In the audo clip, <sound>, what is happening?",
    "Provide a description of the sound. <sound>",
    "Provide a caption for the sound. <sound>",
]

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_caption",
            "description": "Write a caption for an audio clip",
            "parameters": {
                "type": "object",
                "properties": {
                    "caption": {
                        "type": "string",
                    },
                },
                "required": ["caption"],
            },
        },
    }
]


def _build_convo(row) -> List:
    client = openai.Client()

    captions = [row["metadataTags"]]
    sounds = [row["url"]]

    captions_text = "\n".join([f'Tags: "{cap}"' for i, cap in enumerate(captions)])
    prompt = PROMPT.format(captions=captions_text).strip()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": prompt}],
        tools=OPENAI_TOOLS,
        tool_choice={"type": "function", "function": {"name": "write_caption"}},
    )
    resp = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    caption = resp["caption"]

    q = random.choice(PRETRAIN_PHRASES)

    example = {
        "sounds": sounds,
        "messages": [
            {
                "role": ROLE_USER,
                "content": q,
            },
            {
                "role": ROLE_ASSISTANT,
                "content": caption,
            },
        ],
    }
    return example


def main(args):
    data = load_dataset("Chr0my/Epidemic_sounds", split="train")

    os.makedirs(args.cache_folder, exist_ok=True)

    def gen(seeds):
        cache = open(
            os.path.join(args.cache_folder, f"gpt-cache.{seeds[0]}.jsonl"), "a"
        )
        for s in seeds:
            selected_row = data[s]
            try:
                example = _build_convo(selected_row)
                cache.write(json.dumps(example) + "\n")
                yield example
            except Exception as e:
                print(e)
                continue

        cache.close()

    idxs = list(range(len(data)))
    random.shuffle(idxs)

    ds = Dataset.from_generator(
        gen,
        num_proc=args.num_proc,
        gen_kwargs={"seeds": idxs},
    )
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="/data/clap-gpt-pretrain",
    )
    parser.add_argument(
        "-c",
        "--cache_folder",
        type=str,
        default="/data/clap-gpt-pretrain-cache",
    )
    parser.add_argument("-n", "--num_examples", type=int, default=500_000)
    parser.add_argument("-p", "--num_proc", type=int, default=10)
    args = parser.parse_args()
    main(args)
