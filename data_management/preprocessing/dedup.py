import argparse
import json
from hojichar import document_filters, deduplication, Compose, Document
import os
from datetime import datetime


def exec_hojichar_deduplication(lines: list[str], output_base: str, stats: list[dict]):
    remained_lines = []
    cleaner = Compose([
        document_filters.JSONLoader(ignore=True),
        deduplication.GenerateDedupLSH(),
        deduplication.LSHDeduplicator(
            online_dedup=True,
            store_blacklist=True
        ),
        document_filters.JSONDumper()
    ])

    with open(os.path.join(output_base, "result.dedup.jsonl"), "w") as writer:
        with open(os.path.join(output_base, "rejected.dedup.jsonl"), "w") as rejected:
            for line in lines:
                result = cleaner.apply(Document(line))
                if result.is_rejected:
                    rejected.write(result.text + "\n")
                else:
                    writer.write(result.text + "\n")
                    remained_lines.append(result.text)

    with open(os.path.join(output_base, "stat.dedup.jsonl"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")
    stats.append(cleaner.statistics)

    return remained_lines


def dedup_minhashlsh(input_dir: str, output_base: str):
    os.makedirs(output_base, exist_ok=True)
    remained_lines, stats = [], []
    for input_file in os.listdir(input_dir):
        if not input_file.endswith(".jsonl"):
            continue

        with open(os.path.join(input_dir, input_file)) as fp:
            json_lines = fp.readlines()

        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        output_base_for_input: str = os.path.join(output_base, input_file_prefix)
        os.makedirs(output_base_for_input, exist_ok=True)

        remained_lines.append(exec_hojichar_deduplication(
            json_lines, output_base=output_base_for_input, stats=stats))

    with open(os.path.join(output_base, "results.dedup.jsonl"), "w", encoding="utf8") as writer:
        for lines in remained_lines:
            for line in lines:
                writer.write(line + "\n")

    with open(os.path.join(output_base, "stats.dedup.jsonl"), "w", encoding="utf8") as writer:
        for stat in stats:
            writer.write(json.dumps(stat, ensure_ascii=False))
            writer.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./tmp/output")
    args = parser.parse_args()

    start = datetime.now()
    output_base = os.path.join(args.output_dir, start.strftime("%Y%m%d%H%M%S"))

    dedup_minhashlsh(input_dir=args.input_dir, output_base=output_base)


if __name__ == "__main__":
    main()
