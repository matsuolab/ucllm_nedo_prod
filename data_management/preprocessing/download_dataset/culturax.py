import os
import subprocess
import gzip
import logging
import json
import pandas as pd


def __download_config(language: str, index_from: int, index_to: int) -> dict[str, str]:
    if index_to > 1024:
        raise ValueError("index_to must be less than or equal to 1024")
    filebase = "{language}_part_{index}.parquet"
    output_file = "{language}_part_{index_from}-{index_to}.jsonl".format(language=language, 
        index_from=str(index_from).zfill(5), index_to=str(index_to).zfill(5))
    return {"filebase": filebase, "output_file": output_file}


def __execute_download(language: str, download_file: str, output_file_path: str, dataset_root: str) -> None:
    logging.info(f"Downloading {dataset_root}/{download_file}")

    current_dir = os.getcwd()

    # Change directory in order to use git lfs
    os.chdir(dataset_root)
    subprocess.run(["git", "lfs", "pull", "--include", f"{language}/{download_file}"], check=True)
    os.chdir(current_dir)

    logging.info(f"Saving to {output_file_path}")

    df = pd.read_parquet(f"{dataset_root}/{language}/{download_file}")
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        for i, row in df.iterrows():
            try:
                # pandasの行を辞書に変換し、JSON文字列に変換
                json_str = json.dumps(row.to_dict(), ensure_ascii=False)
                output_file.write(json_str + "\n")
            except (TypeError, ValueError) as e:
                logging.info(f"Failed to convert row {i} to JSON: {e}")


def download_dataset(language: str,  output_base: str = "output", index_from: int = 0, index_to: int = 0) -> None:
    """Download the specified C4 dataset from Hugging Face."""
    if index_from < 0:
        raise ValueError("index_from must be greater than or equal to 0")
    if index_to < index_from:
        raise ValueError("index_to must be greater than or equal to index_from")

    # Set the filename and save path based on the dataset name
    dataset = "https://huggingface.co/datasets/uonlp/CulturaX"
    output_path = os.path.join(output_base, "datasets/culturax")
    dataset_root = os.path.join(output_base, "tmp/culturax")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(dataset_root, exist_ok=True)

    # Download file index
    current_dir = os.getcwd()
    print(current_dir)
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

    config = __download_config(language=language, index_from=index_from, index_to=index_to)

    output_file_path = os.path.join(output_path, config["output_file"])
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    for i in range(index_from, index_to+1):
        filename = config["filebase"].format(language=language, index=str(i).zfill(5))
        __execute_download(language=language, download_file=filename,
