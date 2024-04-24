import argparse
from datetime import datetime
import json
from hojichar import document_filters, tokenization, Compose, Document
import os
from tqdm import tqdm
import hojichar
from preprocessing.filtering import custom_token_filters, custom_tokenization, custom_document_filters


def process_json_lines(lines: list[str], output_base: str, output_base_for_input: str, input_file_prefix: str):
    remained_lines = []

    num_cores = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))
    print(f"Using {num_cores} cores for processing.")
    
    cleaner = Compose([
        document_filters.JSONLoader(),
        document_filters.DocumentNormalizer(),
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
        document_filters.JSONDumper(dump_reason=True),
    ])
    input_doc_iter = (Document(line) for line in lines)
    print("start cleaning")
    with hojichar.Parallel(cleaner, num_jobs=num_cores) as pfilter:

        out_doc_iter = pfilter.imap_apply(input_doc_iter)
        out_doc_iter = tqdm(out_doc_iter, desc="data cleaning")

        rejected_filename = f"{input_file_prefix}_rejected_filtering.jsonl"
        #Conveniently read jsonl files directly from the directory
        result_filename = f"{input_file_prefix}_result_filtering.jsonl"
        with open(os.path.join(output_base_for_input, rejected_filename), "w") as rejected:
            with open(os.path.join(output_base, result_filename), "w") as writer:
                for result in out_doc_iter:
                    try:
                        if result.is_rejected:
                            rejected.write(result.text + "\n")
                        else:
                            writer.write(result.text + "\n")
                            #remained_lines.append(result.text)
                    except Exception as e:
                        print(f"Error processing document: {e}")

    with open(os.path.join(output_base_for_input, "stat.filtering.jsonl"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")
    #This will cause oom
    #stats.append(cleaner.statistics)
    #This will cause oom
    #return remained_lines


def __readlines(input_file: str):
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            yield line



def filtering(input_dir: str, output_base: str):
    os.makedirs(output_base, exist_ok=True)

    file_lines = {input_file: __readlines(os.path.join(input_dir, input_file))
                  for input_file in os.listdir(input_dir) if input_file.endswith(".jsonl")}

    stats = []
    for input_file, json_lines in file_lines.items():
        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        output_base_for_input: str = os.path.join(output_base, input_file_prefix)
        os.makedirs(output_base_for_input, exist_ok=True)

        #lines = process_json_lines(json_lines, output_base,output_base_for_input, stats,input_file_prefix)
        process_json_lines(json_lines, output_base,output_base_for_input,input_file_prefix)
        #This will cause oom
        #file_lines[input_file] = lines
"""
    with open(os.path.join(output_base, "results.filtering.jsonl"), "w", encoding="utf8") as writer:
        for _, lines in file_lines.items():
            for line in lines:
                writer.write(line + "\n")

    with open(os.path.join(output_base, "stats.filtering.jsonl"), "w", encoding="utf8") as writer:
        for stat in stats:
            json.dump(stat, writer, ensure_ascii=False)
            writer.write("\n")
"""

def main():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./tmp/output")
    args = parser.parse_args()

    start = datetime.now()
    #output_base = os.path.join(args.output_dir, start.strftime("%Y%m%d%H%M%S"))
    output_base = args.output_dir
    filtering(input_dir=args.input_dir, output_base=output_base)


if __name__ == "__main__":
    main()
