import argparse
from ast import parse
import os
import pathlib

import preprocessing

from preprocessing.download_dataset import c4, wikipedia

ROOT_PATH = pathlib.Path(preprocessing.__path__[0]).resolve().parent
SCRIPT_PATH = os.path.join(ROOT_PATH, "scripts")


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", type=str, help="Dataset to download")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download")
    parser.add_argument("--output_base", type=str, default="output",
                        help="Base directory to save the dataset")
    parser.add_argument("--index_from", type=int, default=0, help="Index to start downloading")
    parser.add_argument("--index_to", type=int, default=0, help="Index to stop downloading")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "c4":
        c4.download_dataset(split=args.split, index_from=args.index_from,
                            index_to=args.index_to, output_base=args.output_base)
    elif args.dataset == "wikipedia":
        wikipedia.download_dataset(date=args.split, output_base=args.output_base)


if __name__ == "__main__":
    main()
