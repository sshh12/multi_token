import argparse

from datasets import load_dataset, concatenate_datasets


def main(args):
    dss = []
    for dataset_path in args.dataset:
        dataset = load_dataset(dataset_path, split="train", data_files="*.arrow")
        dss.append(dataset)

    ds = concatenate_datasets(dss)
    ds = ds.shuffle()
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, action="append")
    parser.add_argument("-o", "--output_folder", type=str)
    args = parser.parse_args()
    main(args)
