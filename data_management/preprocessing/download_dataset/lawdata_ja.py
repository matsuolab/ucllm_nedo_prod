import logging
import os
import requests
import shutil


def download_dataset(output_base: str = "output") -> None:
    filename = f"all_xml.zip"

    dump_path = os.path.join(output_base, f"tmp/lawdata")
    os.makedirs(dump_path, exist_ok=True)

    url = f"https://elaws.e-gov.go.jp/download?file_section=1&only_xml_flag=true"
    if not os.path.exists(os.path.join(dump_path, filename)):
        logging.info(f"Downloading {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            logging.info(f"Saving to {os.path.join(dump_path, filename)}")
            with open(os.path.join(dump_path, filename), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        logging.info(f"File {os.path.join(dump_path, filename)} already exists")
        logging.info(f"Skipping download\n")

    # ダンプデータをパースする
    output_path = os.path.join(output_base, f"lawdata")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    logging.info(f"Parse and process {os.path.join(dump_path, filename)}")
    shutil.unpack_archive(os.path.join(dump_path, filename), output_path)