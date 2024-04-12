
import logging
import os
import subprocess
import preprocessing
import pathlib

ROOT_PATH = pathlib.Path(preprocessing.__path__[0]).resolve().parent
SCRIPT_PATH = os.path.join(ROOT_PATH, "scripts")


def download_dataset(split: str = "", output_base: str = "tmp/output") -> None:
    # Set the filename and save path based on the dataset name
    dataset = "https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt"
    dataset_root = os.path.join(output_base, "tmp/togethercomputer/redpajama")
    os.makedirs(dataset_root, exist_ok=True)

    # Download file index
    if os.path.exists(os.path.join(dataset_root, "urls.txt")):
        logging.info("File index already exists")
        logging.info("Skipping download")
    else:
        current_dir = os.getcwd()
        os.chdir(dataset_root)
        subprocess.call([f"wget {dataset}"], shell=True)
        os.chdir(current_dir)

    # Download the dataset
    output_path = os.path.join(output_base, "datasets/togethercomputer/redpajama")
    os.makedirs(output_path, exist_ok=True)
    subprocess.run([os.path.join(SCRIPT_PATH,  "download_redpajama.sh"), output_path, os.path.join(dataset_root, "urls.txt"), split])
