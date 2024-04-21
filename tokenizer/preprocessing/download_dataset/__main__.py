import argparse
from ast import parse
import os
import pathlib

import preprocessing

from preprocessing.download_dataset import wikipedia
ROOT_PATH = pathlib.Path(preprocessing.__path__[0]).resolve().parent
SCRIPT_PATH = os.path.join(ROOT_PATH, "scripts")


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", type=str, help="Dataset to download")
    parser.add_argument("--split", type=str, default="", help="Dataset split to download")
    parser.add_argument("--output_base", type=str, default="./tmp/output",
                        help="Base directory to save the dataset")
    parser.add_argument("--language", type=str, help="Language to download")


    return parser.parse_args()


def main():
    args = parse_args()
    wikipedia.download_dataset(date=args.split,   
        output_base=args.output_base,
        lang=args.language)

if __name__ == "__main__":
    main()
