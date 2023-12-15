from typing import List
import random
import argparse

from datasets import load_dataset
from datasets import Dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER

DATASET_ARGS = dict(
    path="mozilla-foundation/common_voice_15_0", name="en", split="train"
)

PRETRAIN_PHRASES = [
    "Repeat the content of the audio <speech>",
    "Transcribe <speech>",
    "What is being said in <speech>",
    "Can you interpret <speech>?",
    "Please convert <speech> into text",
    "What does <speech> say?",
    "Could you transcribe <speech> for me?",
    "I need the text of <speech>",
    "Can you write out <speech>?",
    "What's the content of <speech>?",
    "Please provide the transcript of <speech>",
    "Can you decode <speech>?",
    "What is the transcription of <speech>?",
    "Can you jot down <speech>?",
    "What is the written form of <speech>?",
    "Can you scribe <speech>?",
]


def _write_convo(idx, row) -> List:
    example = {
        "speech_audios": [{"dataset_args": DATASET_ARGS, "idx": idx}],
    }
    phrase = random.choice(PRETRAIN_PHRASES)
    example["messages"] = [
        {
            "role": ROLE_USER,
            "content": phrase,
        },
        {
            "role": ROLE_ASSISTANT,
            "content": row["text"] if "text" in row else row["sentence"],
        },
    ]
    return example


def main(args):
    audio_dataset = load_dataset(**DATASET_ARGS)

    def gen():
        i = 0
        idxes = list(range(len(audio_dataset)))
        random.shuffle(idxes)
        for k in idxes:
            try:
                yield _write_convo(k, audio_dataset[k])
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
    parser.add_argument("-n", "--max_examples", type=int, default=200_000)
    args = parser.parse_args()
    main(args)
