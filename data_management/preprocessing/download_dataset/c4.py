import os
import subprocess
import gzip
import logging
import json


def __download_config(split: str, index_from: int, index_to: int) -> dict[str, str]:
    if split == "train":
        if index_to > 1024:
            raise ValueError("index_to must be less than or equal to 1024")
        filebase = "c4-ja.tfrecord-{index}-of-01024.json.gz"
        output_file = "c4-ja_{index_from}-{index_to}.jsonl".format(
            index_from=str(index_from).zfill(5), index_to=str(index_to).zfill(5))
    elif split == "valid":
        if index_to > 8:
            raise ValueError("index_to must be less than or equal to 8")
        filebase = "c4-ja-validation.tfrecord-{index}-of-00008.json.gz"
        output_file = "c4-ja-validation_{index_from}-{index_to}.jsonl".format(
            index_from=str(index_from).zfill(5), index_to=str(index_to).zfill(5))
    return {"filebase": filebase, "output_file": output_file}


def __execute_download(download_file: str, output_file_path: str, dataset_root: str) -> None:
    logging.info(f"Downloading {dataset_root}/{download_file}")

    current_dir = os.getcwd()

    # Change directory in order to use git lfs
    os.chdir(dataset_root)
    subprocess.run(["git", "lfs", "pull", "--include", f"multilingual/{download_file}"], check=True)
    os.chdir(current_dir)

    logging.info(f"Saving to {output_file_path}")
    with gzip.open(f"{dataset_root}/multilingual/{download_file}", 'rb') as input_file:
        content = input_file.read().decode("utf-8")

    with open(output_file_path, 'a', encoding="utf-8") as output_file:
        for i, line in enumerate(content.split("\n")):
            try:
                line = json.loads(line)
                output_file.write(json.dumps(line, ensure_ascii=False) + "\n")
            except (json.JSONDecodeError) as e:
                logging.info(f"Failed to decode line {i}: {e}")



def download_dataset(split: str, output_base: str = "output", index_from: int = 0, index_to: int = 0) -> None:
    """Download the specified C4 dataset from Hugging Face."""
    if index_from < 0:
        raise ValueError("index_from must be greater than or equal to 0")
    if index_to < index_from:
        raise ValueError("index_to must be greater than or equal to index_from")

    # Set the filename and save path based on the dataset name
    dataset = "https://huggingface.co/datasets/allenai/c4"
    output_path = os.path.join(output_base, "datasets/allenai/c4")
    dataset_root = os.path.join(output_base, "tmp/allenai/c4")
    os.makedirs(output_path, exist_ok=True)
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

    config = __download_config(split=split, index_from=index_from, index_to=index_to)

    output_file_path = os.path.join(output_path, config["output_file"])
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    for i in range(index_from, index_to):
        filename = config["filebase"].format(index=str(i).zfill(5))
        __execute_download(download_file=filename,
                           output_file_path=output_file_path, dataset_root=dataset_root)
