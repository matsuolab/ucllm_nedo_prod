# Appends a path to import python scripts that are in other directories.
import os
import sys

import argparse
import sentencepiece as spm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "word", "char"])
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--train_extremely_large_corpus", type=bool, default=True)
    parser.add_argument("--pretokenization_delimiter", type=str, default="")
    parser.add_argument("--max_sentencepiece_length", type=int, default=8)
        
    args = parser.parse_args()
    print(f"{args = }")
    return args


def main():
    args = parse_arguments()

    # Trains a SentencePiece tokenizer. After training, *.model and *.vocab will be saved in the current directory.
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        num_threads=args.num_threads,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
        normalization_rule_name='identity',
        pretokenization_delimiter=args.pretokenization_delimiter,
        user_defined_symbols=['\n'],
        max_sentencepiece_length=args.max_sentencepiece_length, # 日本語は最大長8
        byte_fallback=True,
        split_digits=True,
        split_by_whitespace=True, # モデル作成時は空白で区切る
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
    )


if __name__ == "__main__":
    main()
