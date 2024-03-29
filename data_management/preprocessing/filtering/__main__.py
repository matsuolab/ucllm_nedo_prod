import argparse
from datetime import datetime
import json
from hojichar import document_filters, tokenization, Compose, Document
import os

from preprocessing.filtering import custom_token_filters, custom_tokenization, custom_document_filters


def process_json_lines(lines: list[str], output_base: str, stats: list[dict]):
    remained_lines = []
    cleaner = Compose([
        document_filters.JSONLoader(),
        document_filters.DocumentNormalizer(),
        document_filters.DocumentLengthFilter(min_doc_len=20,max_doc_len=1000),
        document_filters.DiscardBBSComments(),
        document_filters.DiscardAds(),
        document_filters.DiscardDiscriminationContentJa(),
        custom_document_filters.DiscardAdultContentJa(),
        custom_tokenization.NewLineSentenceTokenizer(),
        custom_token_filters.RemoveOneword(),
        custom_tokenization.MergeTokens(delimiter="\n"),
        custom_tokenization.WakatiTokenizer(),
        custom_token_filters.RemoveDate(),
        tokenization.MergeTokens(),
        document_filters.MaskPersonalInformation(),
        document_filters.JSONDumper(dump_reason=False),
    ])

    with open(os.path.join(output_base, "rejected.filtering.jsonl"), "w") as rejected:
        with open(os.path.join(output_base, "result.filtering.jsonl"), "w") as writer:
            for line in lines:
                result = cleaner.apply(Document(line))
                if result.is_rejected:
                    rejected.write(result.text + "\n")
                else:
                    writer.write(result.text + "\n")
                    remained_lines.append(result.text)

    with open(os.path.join(output_base, "stat.filtering.jsonl"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")

    stats.append(cleaner.statistics)

    return remained_lines


def __readlines(input_file: str):
    with open(input_file) as fp:
        return fp.readlines()


def filtering(input_dir: str, output_base: str):
    os.makedirs(output_base, exist_ok=True)

    file_lines = {input_file: __readlines(os.path.join(input_dir, input_file))
                  for input_file in os.listdir(input_dir) if input_file.endswith(".jsonl")}

    stats = []
    for input_file, json_lines in file_lines.items():
        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        output_base_for_input: str = os.path.join(output_base, input_file_prefix)
        os.makedirs(output_base_for_input, exist_ok=True)

        lines = process_json_lines(json_lines, output_base_for_input, stats)
        file_lines[input_file] = lines

    with open(os.path.join(output_base, "results.filtering.jsonl"), "w", encoding="utf8") as writer:
        for _, lines in file_lines.items():
            for line in lines:
                writer.write(line + "\n")

    with open(os.path.join(output_base, "stats.filtering.jsonl"), "w", encoding="utf8") as writer:
        for stat in stats:
            json.dump(stat, writer, ensure_ascii=False)
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

    filtering(input_dir=args.input_dir, output_base=output_base)


if __name__ == "__main__":
    main()
