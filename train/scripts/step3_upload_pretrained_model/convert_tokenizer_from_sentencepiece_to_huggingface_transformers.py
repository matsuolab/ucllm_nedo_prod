# Appends a path to import python scripts that are in other directories.
import os
import sys
sys.path.append(os.path.join(os.environ["HOME"], "ucllm_nedo_dev/train/scripts/common/"))


import argparse
from transformers import T5Tokenizer
from special_token_list import UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, EOD_TOKEN, MASK_TOKEN


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_file", type=str, required=True)
    parser.add_argument("--output_tokenizer_dir", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def main() -> None:
    args = parse_arguments()

    # Converts the tokenizer from SentencePiece format to HuggingFace Transformers format by loading with `T5Tokenizer`.
    # Note: `PreTrainedTokenizerFast` (base class) doesn't support byte fallback, but `T5Tokenizer` (derived class) supports byte fallback
    # https://zenn.dev/selllous/articles/transformers_pretrain_to_ft#tokenizers-t5tokenizer%E5%BD%A2%E5%BC%8F%E3%81%B8%E3%81%AE%E5%A4%89%E6%8F%9B
    output_tokenizer = T5Tokenizer(
        vocab_file=args.input_tokenizer_file,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=CLS_TOKEN,
        mask_token=MASK_TOKEN,
        additional_special_tokens=[
            EOD_TOKEN,
        ],  # Note: `NEWLINE_TOKEN` is NOT needed in `additional_special_tokens`.
        extra_ids=0,
        model_max_length=2048,  # TODO: Remove hard coding and/or magic number.
        split_special_tokens=True,
    )

    os.makedirs(args.output_tokenizer_dir, exist_ok=True)
    output_tokenizer.save_pretrained(args.output_tokenizer_dir)


if __name__ == "__main__":
    main()
