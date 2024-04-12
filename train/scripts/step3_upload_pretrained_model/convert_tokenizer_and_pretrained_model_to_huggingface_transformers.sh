#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_dev/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"
echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
input_tokenizer_file=""
input_model_dir=""
output_tokenizer_and_model_dir=""

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --input_model_dir) input_model_dir=${2}; shift ;;
        --output_tokenizer_and_model_dir) output_tokenizer_and_model_dir=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]] || [[ -z ${input_model_dir} ]] || [[ -z ${output_tokenizer_and_model_dir} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file> --input_model_dir <input_model_dir> --output_tokenizer_and_model_dir <output_tokenizer_and_model_dir>"
    exit 1
fi

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_model_dir = ${input_model_dir}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}

# Converts the tokenizer from SentencePiece format to HuggingFace Transformers format.
python ${ucllm_nedo_dev_train_dir}/scripts/step3_upload_pretrained_model/convert_tokenizer_from_sentencepiece_to_huggingface_transformers.py \
    --input_tokenizer_file ${input_tokenizer_file} \
    --output_tokenizer_dir ${output_tokenizer_and_model_dir}

# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
python ${megatron_deepspeed_dir}/tools/convert_checkpoint/deepspeed_to_transformers.py \
    --input_folder ${input_model_dir} \
    --output_folder ${output_tokenizer_and_model_dir}

echo ""
echo "Finished converting the tokenizer and the pretrained model to HuggingFace Transformers format."
echo ""
