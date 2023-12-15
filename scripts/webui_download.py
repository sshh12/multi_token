from huggingface_hub import snapshot_download
import argparse
import os
import glob
import tqdm


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dl_path = snapshot_download(repo_id="biglab/webui-all", repo_type="dataset")

    combined_zip_path = os.path.join(output_dir, "webui-merged.zip")
    if not os.path.exists(combined_zip_path):
        part_paths = sorted(glob.glob(os.path.join(dl_path, "*.zip.*")))
        print("Merging...", len(part_paths), "parts")
        with open(combined_zip_path, "wb") as merged_fp:
            for fn in tqdm.tqdm(part_paths):
                with open(fn, "rb") as part_fp:
                    merged_fp.write(part_fp.read())
    print(combined_zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/webui")
    args = parser.parse_args()
    main(args.output_dir)
