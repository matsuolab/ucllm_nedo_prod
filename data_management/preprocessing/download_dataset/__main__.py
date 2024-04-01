import argparse
from ast import parse
import os
import pathlib

import preprocessing

from preprocessing.download_dataset import c4, wikipedia, redpajama, redpajama_v2, lawdata_ja

ROOT_PATH = pathlib.Path(preprocessing.__path__[0]).resolve().parent
SCRIPT_PATH = os.path.join(ROOT_PATH, "scripts")


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", type=str, help="Dataset to download")
    parser.add_argument("--split", type=str, default="", help="Dataset split to download")
    parser.add_argument("--output_base", type=str, default="./tmp/output",
                        help="Base directory to save the dataset")
    parser.add_argument("--index_from", type=int, default=0, help="Index to start downloading")
    parser.add_argument("--index_to", type=int, default=0, help="Index to stop downloading")
    parser.add_argument("--snapshot", type=str, help="CC snapshot to download")
    parser.add_argument("--language", type=str, help="Language to download")
    parser.add_argument("--partition", type=str, help="Partition to download")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "c4":
        c4.download_dataset(split=args.split or "train", index_from=args.index_from,
                            index_to=args.index_to, output_base=args.output_base)
    elif args.dataset == "wikipedia":
        wikipedia.download_dataset(date=args.split, output_base=args.output_base)
    elif args.dataset == "redpajama":
        redpajama.download_dataset(split=args.split, output_base=args.output_base)
    elif args.dataset == "redpajama_v2":
        redpajama_v2.download_dataset(snapshot=args.snapshot, language=args.language, partition=args.partition, output_base=args.output_base)
    elif args.dataset == "lawdata":
        lawdata_ja.download_dataset(output_base=args.output_base)


if __name__ == "__main__":
    main()
