import argparse
import logging
import os
import shutil
import subprocess


def __execute_download(download_file: str, output_path: str, dataset_root: str) -> None:
    logging.info(f"Downloading {download_file} to {dataset_root}")
    current_dir = os.getcwd()

    # Change directory in order to use git lfs
    os.chdir(dataset_root)
    logging.info(f"Pulling {download_file} from git lfs")
    subprocess.run(["git", "lfs", "pull", "--include", download_file], check=True)
    os.chdir(current_dir)

    logging.info(f"Copying {download_file} to {output_path}")
    shutil.copy(os.path.join(dataset_root, download_file), os.path.join(output_path, download_file))


def download_dataset(output_base: str = "output") -> None:
    # Set the filename and save path based on the dataset name
    dataset = "https://huggingface.co/datasets/taka-yayoi/databricks-dolly-15k-ja"
    dataset_root = os.path.join(output_base, "tmp/taka-yayoi/databricks-dolly-15k-ja")
    os.makedirs(dataset_root, exist_ok=True)

    # Download file index
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(dataset_root, ".git")):
        # Change directory in order to use git lfs

        os.chdir(dataset_root)
        subprocess.call(
            [f"GIT_LFS_SKIP_SMUDGE=1 git pull {dataset}"], shell=True)
    else:
        subprocess.call(
            [f"GIT_LFS_SKIP_SMUDGE=1 git clone {dataset} {dataset_root}"], shell=True)
        os.chdir(dataset_root)
        subprocess.call(
            [f"git lfs install"], shell=True)
    os.chdir(current_dir)

    output_path = os.path.join(output_base, "datasets/databricks-dolly-15k-ja")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    filename = "databricks_dolly_15k_ja_for_dolly_training.jsonl"
    __execute_download(download_file=filename,
                       output_path=output_path, dataset_root=dataset_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--output_base", type=str, default="output",
                        help="Base directory to save the dataset")

    return parser.parse_args()


def main():
    args = parse_args()
    download_dataset(output_base=args.output_base)


if __name__ == "__main__":
    main()
