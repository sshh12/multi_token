import argparse
import shutil
import os

from huggingface_hub import HfApi

USEFUL_FILES = [
    "adapter_config.json",
    "adapter_model.bin",
    "config.json",
    "non_lora_trainables.bin",
    "README.md",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "trainer_state.json",
    "model_named_parameters.txt",
]


def main(args):
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, repo_type="model")

    checkpoints = [fn for fn in os.listdir(args.model_folder) if fn.startswith("check")]
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

    if (
        not os.path.exists(os.path.join(args.model_folder, "config.json"))
        and len(checkpoints) > 0
    ):
        last_checkpoint = os.path.join(args.model_folder, checkpoints[-1])
        for fn in USEFUL_FILES:
            checkpoint_fn = os.path.join(last_checkpoint, fn)
            new_fn = os.path.join(args.model_folder, fn)
            if os.path.exists(checkpoint_fn) and not os.path.exists(new_fn):
                shutil.copy(checkpoint_fn, args.model_folder)

    api.upload_folder(
        repo_id=args.repo, allow_patterns=USEFUL_FILES, folder_path=args.model_folder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo", type=str)
    parser.add_argument("-m", "--model_folder", type=str)
    args = parser.parse_args()
    main(args)
