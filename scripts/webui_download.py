from huggingface_hub import snapshot_download
import argparse
import subprocess
import os
import glob
from zipfile import ZipFile


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dl_path = snapshot_download(repo_id="biglab/webui-all", repo_type="dataset")

    combined_zip_path = os.path.join(output_dir, "webui-all-full.zip")
    if not os.path.exists(combined_zip_path):
        part_paths = sorted(glob.glob(os.path.join(dl_path, "*.zip.*")))
        print("Merging...", len(part_paths), "parts")
        subprocess.check_output(
            [
                "sh",
                "-c",
                "cat " + " ".join(part_paths) + " > " + combined_zip_path,
            ]
        )
    print(combined_zip_path)
    with ZipFile(combined_zip_path) as myzip:
        # show files
        print(myzip.namelist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/webui")
    args = parser.parse_args()
    main(args.output_dir)
