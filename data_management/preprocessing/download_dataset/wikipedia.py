import logging
import os
import bz2
import json
import shutil
import mwxml
import hashlib
import requests


NUM_FILES = os.environ.get('NUM_FILES', 100)


def process_dump(page, path, file_index):
    with open(os.path.join(path, f"{file_index}.jsonl"), 'a', encoding='utf-8') as output_file:
        id = page.id
        title = page.title
        for revision in page:
            text = revision.text
            if text:
                break

        # 記事のタイトルとテキストをJSON形式に変換する
        article_json = json.dumps({
            'id': id,
            'title': title,
            'text': text,
        }, ensure_ascii=False)

        # JSONをファイルに書き込む
        output_file.write(article_json + '\n')


def download_dataset(date: str, output_base: str = "output", lang: str = "is") -> None:
    filename = f"{lang}wiki-{date}-pages-articles-multistream.xml.bz2"

    dump_path = os.path.join(output_base, f"tmp/wikipedia/{date}/{lang}")
    os.makedirs(dump_path, exist_ok=True)

    url = f"https://dumps.wikimedia.org/{lang}wiki/{date}/{filename}"
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
    output_path = os.path.join(output_base, f"datasets/wikipedia/{date}/{lang}")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    logging.info(f"Parse and process {os.path.join(dump_path, filename)}")
    with bz2.open(os.path.join(dump_path, filename), 'rt') as compressed_file:
        dump = mwxml.Dump.from_file(compressed_file)
        for index, page in enumerate(dump):
            if index > 0 and index % 1000 == 0:
                logging.info(f"Processed {index} articles")

            if page.namespace == 0 and page.redirect is None:
                file_index = int(hashlib.sha256(page.title.encode()).hexdigest(), 16) % NUM_FILES
                process_dump(page, output_path, file_index)
