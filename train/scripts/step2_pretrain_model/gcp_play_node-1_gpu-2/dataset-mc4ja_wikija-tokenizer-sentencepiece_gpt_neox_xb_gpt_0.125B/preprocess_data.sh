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
input_file_dir=""
output_file_dir=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_tokenizer_file) input_tokenizer_file="$2"; shift ;;
        --input_file_dir) input_file_dir="$2"; shift ;;
        --output_file_dir) output_file_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]] || [[ -z ${input_file_dir} ]] || [[ -z ${output_file_dir} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file> --input_file_dir <input_file_dir> --output_file_dir <output_file_dir>"
    exit 1
fi

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_file_dir = ${input_file_dir}"
echo "output_file_dir = ${output_file_dir}"
echo ""



# Loops through each .jsonl file in the input_file_dir and processes it.
for file in ${input_file_dir}/*.jsonl; do
    # Sets the output file path without the extension
    output_file="${output_file_dir}/$(basename "${file}" .jsonl)"
    
    # Checks if the output files already exist
    if [ ! -f "${output_file}_text_document.bin" ] || [ ! -f "${output_file}_text_document.idx" ]; then
        echo "Processing ${file}..."
        python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
            --tokenizer-type SentencePieceTokenizer \
            --tokenizer-model ${input_tokenizer_file} \
            --input ${file} \
            --output-prefix ${output_file} \
            --dataset-impl mmap \
            --workers $(grep -c ^processor /proc/cpuinfo) \
            --append-eod
    else
        echo "Both ${output_file}_text_document.bin and ${output_file}_text_document.idx already exist."
    fi
done
