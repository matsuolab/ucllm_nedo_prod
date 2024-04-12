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
input_model_name_or_path=""
output_tokenizer_and_model_dir=""
wandb_entity=""
wandb_project=""

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_model_name_or_path) input_model_name_or_path=${2}; shift ;;
        --output_tokenizer_and_model_dir) output_tokenizer_and_model_dir=${2}; shift ;;
        --wandb_entity) wandb_entity=${2}; shift ;;
        --wandb_project) wandb_project=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_model_name_or_path} ]] || [[ -z ${output_tokenizer_and_model_dir} ]] || [[ -z ${wandb_entity} ]] || [[ -z ${wandb_project} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_model_name_or_path <input_model_name_or_path> --output_tokenizer_and_model_dir <output_tokenizer_and_model_dir> --wandb_entity <wandb_entity> --wandb_project <wandb_project>"
    exit 1
fi

# Prints the arguments.
echo "input_model_name_or_path = ${input_model_name_or_path}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo "wandb_entity = ${wandb_entity}"
echo "wandb_project = ${wandb_project}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}

# If openassistant_best_replies_train.jsonl doesn't exist yet,
# then downloads openassistant_best_replies_train.jsonl.
dataset_file=${ucllm_nedo_dev_train_dir}/llm-jp-sft/dataset/openassistant_best_replies_train.jsonl
if [ ! -f ${dataset_file} ]; then
    echo "${dataset_file} doesn't exist yet, so download arxiv.jsonl and preprocess the data."
    wget https://huggingface.co/datasets/timdettmers/openassistant-guanaco/resolve/main/openassistant_best_replies_train.jsonl \
        --directory-prefix ${ucllm_nedo_dev_train_dir}/llm-jp-sft/dataset/
else
    echo "${dataset_file} already exists."
fi
echo ""

# Logging.
log_path="${output_tokenizer_and_model_dir}/log"
mkdir -p ${log_path}
host="${HOSTNAME}"
current_time=$(date "+%Y.%m.%d_%H.%M.%S")

# Extracts `input_model_name` form `input_model_name_or_path`.
input_model_name="${input_model_name_or_path}"
input_model_name="${input_model_name%/}"  # Removes a trailing slash "/" if it exists.
input_model_name="${input_model_name##*/}"  # Extracts the substring after the last "/".

# W&B options.
# Note: the W&B options use "underscore" (like `--use_wandb`) instead of "hyphen" (like `--use-wandb`).
# Note: the W&B group name `--wandb_group` should be less than 128 characters.
wandb_options=" \
    --use_wandb \
    --wandb_entity ${wandb_entity} \
    --wandb_project ${wandb_project} \
    --wandb_group finetune_${input_model_name}_${host}_${current_time}"

# Creates a hostfile.
script_dir=$(dirname "$0")
hostfile="${script_dir}/hostfile_jobid-${JOB_ID}"
while read -r line
do
  echo "${line} slots=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
done < "${SGE_JOB_HOSTLIST}" > "${hostfile}"
echo "hostfile = ${hostfile}"
cat ${hostfile}
echo ""

# Finetunes the pretrained model.
deepspeed --hostfile ${hostfile} \
    ${ucllm_nedo_dev_train_dir}/llm-jp-sft/train.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 2048 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --data_files ${dataset_file} \
    --model_name_or_path ${input_model_name_or_path} \
    --output_dir ${output_tokenizer_and_model_dir} \
    --use_fast False \
    --instruction_template "### Human:" \
    --response_template "### Assistant:" \
    --deepspeed ${script_dir}/deepspeed_config/ds_config_zero3.json \
    ${wandb_options} \
    2>&1 | tee ${log_path}/${host}_${current_time}.log

echo ""
echo "Finished finetuning the pretrained model."
echo ""
