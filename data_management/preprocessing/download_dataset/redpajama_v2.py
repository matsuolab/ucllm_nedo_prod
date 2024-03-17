import logging
import os
import subprocess
import preprocessing
import pathlib

ROOT_PATH = pathlib.Path(preprocessing.__path__[0]).resolve().parent
SCRIPT_PATH = os.path.join(ROOT_PATH, "scripts")


def download_dataset(snapshot: str, language: str, partition: str, components: list[str] = [], output_base: str = "tmp/output") -> None:
    # Set the filename and save path based on the dataset name
    dataset_root = os.path.join(output_base, "tmp/togethercomputer/redpajama-v2")
    os.makedirs(dataset_root, exist_ok=True)

    base_url = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
    listings_tag = f"{language}-{snapshot}-{partition}"
    listings_file = f"{dataset_root}/{listings_tag}.txt"

    # Download file index
    if os.path.exists(listings_file):
        logging.info("File index already exists")
        logging.info("Skipping download")
    else:
        listings_endpoint = f"{base_url}/listings/{listings_tag}.txt"
        current_dir = os.getcwd()
        os.chdir(dataset_root)
        subprocess.call([f"wget {listings_endpoint}"], shell=True)
        os.chdir(current_dir)

    # Download the dataset
    output_path = os.path.join(output_base, "datasets/togethercomputer/redpajama-v2")
    os.makedirs(output_path, exist_ok=True)

    subprocess.run([os.path.join(SCRIPT_PATH,  "download_redpajama_v2.sh"), listings_file, output_path])
