# Appends a path to import python scripts that are in other directories.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../common/'))

import argparse
import json
import sentencepiece as spm
from transformers import T5Tokenizer
from special_token_list import UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, EOD_TOKEN, MASK_TOKEN


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_file", type=str, required=True)
    parser.add_argument("--input_model_max_length", type=int, required=True)
    parser.add_argument("--output_tokenizer_dir", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def convert_tokenizer(input_tokenizer_file, input_model_max_length):
    # Converts the tokenizer from SentencePiece format to HuggingFace Transformers format by loading with `T5Tokenizer`.
    # Note: `PreTrainedTokenizer` (base class) doesn't support byte fallback, but `T5Tokenizer` (derived class) supports byte fallback.
    # https://zenn.dev/selllous/articles/transformers_pretrain_to_ft#tokenizers-t5tokenizer%E5%BD%A2%E5%BC%8F%E3%81%B8%E3%81%AE%E5%A4%89%E6%8F%9B
    # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/tokenization_utils_base.py#L823-L832
    output_tokenizer = T5Tokenizer(
        vocab_file=input_tokenizer_file,
        model_max_length=input_model_max_length,
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
        split_special_tokens=True,
        extra_ids=0,
    )

    return output_tokenizer


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def pass_or_fail(lhs, rhs) -> str:
    return "Pass" if lhs == rhs else "Fail"


def compare_tokenizer(sentencepiece_tokenizer, huggingface_tokenizer, test_text: str) -> None:
    print(f"==================================================================")
    print()
    print(f"input text: {dumps(test_text)}")
    print()

    encoded_text_sentencepiece = sentencepiece_tokenizer.encode(test_text, out_type="immutable_proto")

    encoded_pieces_sentencepiece = [piece.piece for piece in encoded_text_sentencepiece.pieces]
    encoded_pieces_huggingface = huggingface_tokenizer.tokenize(test_text)
    print(f"encoded pieces")
    print(f"        sp: {dumps(encoded_pieces_sentencepiece)}")
    print(f"-> {pass_or_fail(encoded_pieces_sentencepiece, encoded_pieces_huggingface)} hf: {dumps(encoded_pieces_huggingface)}")
    print()

    encoded_ids_sentencepiece = [piece.id for piece in encoded_text_sentencepiece.pieces]
    encoded_ids_huggingface = huggingface_tokenizer.encode(test_text, add_special_tokens=False)
    print(f"encoded ids")
    print(f"        sp: {dumps(encoded_ids_sentencepiece)}")
    print(f"-> {pass_or_fail(encoded_ids_sentencepiece, encoded_ids_huggingface)} hf: {dumps(encoded_ids_huggingface)}")
    print()

    dencoded_text_sentencepiece = sentencepiece_tokenizer.decode(encoded_ids_sentencepiece)
    dencoded_text_huggingface = huggingface_tokenizer.decode(encoded_ids_huggingface)
    print(f"decoded text")
    print(f"-> {pass_or_fail(dencoded_text_sentencepiece, test_text)} sp: {dumps(dencoded_text_sentencepiece)}")
    print(f"-> {pass_or_fail(dencoded_text_huggingface, test_text)} hf: {dumps(dencoded_text_huggingface)}")
    print()


def main() -> None:
    args = parse_arguments()

    print("Converting the sentencepiece tokenizer to the huggingface tokenizer...")
    output_tokenizer = convert_tokenizer(args.input_tokenizer_file, args.input_model_max_length)
    os.makedirs(args.output_tokenizer_dir, exist_ok=True)
    output_tokenizer.save_pretrained(args.output_tokenizer_dir)

    print("Comparing the sentencepiece tokenizer and the huggingface tokenizer...")
    input_tokenizer = spm.SentencePieceProcessor(model_file=args.input_tokenizer_file)
    test_texts = [
        "### Test for Japanese: 大嘗祭は、皇室行事。",
        "### Test for special token: <s> </s> <pad> <CLS> <SEP> <EOD> <MASK>",
        "### Test for newline at the middle of sentence: \n 1 newline. \n\n 2 newlines. \n\n\n 3 newlines. \n\n\n\n 4 newlines.",
        "### Test for whitespace at the middle of sentence: 1 space.  2 spaces.   3 spaces.    4 spaces.",
    ]
    for test_text in test_texts:
        compare_tokenizer(input_tokenizer, output_tokenizer, test_text)


if __name__ == "__main__":
    main()
