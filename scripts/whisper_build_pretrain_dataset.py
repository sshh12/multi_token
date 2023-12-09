from typing import List
import random
import argparse

from datasets import load_dataset
from datasets import Dataset

from multi_token.constants import ROLE_ASSISTANT, ROLE_USER

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


def _write_convo(row) -> List:
    audio = dict(row["audio"])
    audio.pop("path")
    example = {
        "speech_audios": [audio],
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
    audio_dataset = load_dataset(
        "mozilla-foundation/common_voice_15_0",
        "en",
        split="train",
        streaming=True,
    )

    def gen():
        i = 0
        for row in audio_dataset:
            try:
                yield _write_convo(row)
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
