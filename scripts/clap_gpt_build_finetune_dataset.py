from typing import List
import argparse
import json
import os
import random
import openai

from datasets import Dataset, load_dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER

PROMPT = """
You are helping train a sound assistant that can take audio inputs and output text.

You can hear an audio file with the following metadata tags:
{captions}

{question}

Include the question and answer.
"""

QUESTIONS = [
    "Ask a question about the content of the audio.",
    "Ask a complex question about the content of the audio.",
    "Ask a complex question that is relevant to the content of the audio, for example, asking about background knowledge of the things mentioned. Do not ask about uncertain details.",
    "Ask a complex question that is relevant to the content of the audio, for example, asking about the events referred to in the audio. Do not ask about uncertain details.",
    "Ask about your thoughts on the audio.",
    "Ask about what occurs in the audio.",
    "Ask a question on a topic that related to the audio.",
    "Ask a question that classifies the audio in some way.",
    "Ask a question that can only be answered by listening to the audio.",
]


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_chat",
            "description": "Create a training example",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question, must be provided",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer to the question, must be provided",
                    },
                },
                "required": ["question", "answer"],
            },
        },
    }
]


def _build_convo(row) -> List:
    client = openai.Client()

    captions = [row["metadataTags"]]
    paths = [row["url"]]

    captions_text = "\n".join([f"{cap}" for i, cap in enumerate(captions)])
    prompt = PROMPT.format(
        captions=captions_text, question=random.choice(QUESTIONS)
    ).strip()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": prompt}],
        tools=OPENAI_TOOLS,
        tool_choice={"type": "function", "function": {"name": "create_chat"}},
    )
    resp = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    if "answer" not in resp:
        print(resp)
    q = resp["question"]
    a = resp["answer"]

    if random.choice([True, False]):
        q = "<sound>" * len(captions) + " " + q
    else:
        q = q + " " + "<sound>" * len(captions)

    example = {
        "sounds": paths,
        "messages": [
            {
                "role": ROLE_USER,
                "content": q,
            },
            {
                "role": ROLE_ASSISTANT,
                "content": a,
            },
        ],
    }
    return example


def main(args):
    data = load_dataset("Chr0my/Epidemic_sounds", split="train")
    data_idxs = list(range(len(data)))

    os.makedirs(args.cache_folder, exist_ok=True)

    def gen(seeds):
        r = random.Random(seeds[0] + 3)
        cache = open(
            os.path.join(args.cache_folder, f"gpt-cache.{seeds[0]}.jsonl"), "a"
        )
        i = 0
        while i < len(seeds):
            selected_idxs = r.sample(data_idxs, k=1)[0]
            selected_example = data[selected_idxs]
            try:
                example = _build_convo(selected_example)
                cache.write(json.dumps(example) + "\n")
                yield example
                i += 1
            except Exception as e:
                print(e)
                continue
        cache.close()

    ds = Dataset.from_generator(
        gen,
        num_proc=args.num_proc,
        gen_kwargs={"seeds": list(range(args.num_examples))},
    )
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="/data/clap-gpt-finetune",
    )
    parser.add_argument(
        "-c",
        "--cache_folder",
        type=str,
        default="/data/clap-gpt-finetune-cache",
    )
    parser.add_argument("-n", "--num_examples", type=int, default=100_000)
    parser.add_argument("-p", "--num_proc", type=int, default=10)
    args = parser.parse_args()
    main(args)
