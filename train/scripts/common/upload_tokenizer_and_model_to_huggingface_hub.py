import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_and_model_dir", type=str, required=True)
    parser.add_argument("--output_model_name", type=str, required=True)
    parser.add_argument("--test_prompt_text", type=str, default="Once upon a time,")
    args = parser.parse_args()
    print(f"{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(input_tokenizer_and_model_dir, device_map="auto")
    return tokenizer, model


def test_tokenizer_and_model(tokenizer, model, prompt_text: str) -> str:
    encoded_prompt_text = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        encoded_generation_text = model.generate(encoded_prompt_text, max_new_tokens=50)[0]
    decoded_generation_text = tokenizer.decode(encoded_generation_text)
    return decoded_generation_text


def main() -> None:
    args = parse_arguments()

    # Loads and tests the tokenizer and the model.
    tokenizer, model = load_tokenizer_and_model(args.input_tokenizer_and_model_dir)
    decoded_generation_text = test_tokenizer_and_model(tokenizer, model, args.test_prompt_text)

    # Checks the generated text briefly.
    print()
    print(f"{args.test_prompt_text = }")
    print(f"{decoded_generation_text = }")
    print()
    if len(decoded_generation_text) <= len(args.test_prompt_text):
        print("Error: The generated text should not be shorter than the prompt text."
              " Something went wrong, so please check either the tokenizer or the model."
              " This program will exit without uploading the tokenizer and the model to HuggingFace Hub.")
        return

    # Uploads the tokenizer and the model to HuggingFace Hub.
    tokenizer.push_to_hub(args.output_model_name)
    model.push_to_hub(args.output_model_name)


if __name__ == "__main__":
    main()
