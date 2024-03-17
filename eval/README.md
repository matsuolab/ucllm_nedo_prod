## 評価ステップの手順

### 環境構築

```
git clone git@github.com:matsuolab/llm-leaderboard.git
pip3 install -r llm-leaderboard/requirements.txt

export LANG=ja_JP.UTF-8
# https://wandb.ai/settings#api
export WANDB_API_KEY=<your WANDB_API_KEY>
# 運営から共有されたkeyを指定してください
export OPENAI_API_KEY=<your OPENAI_API_KEY>
# if needed, please login in huggingface(private設定にしている際など)
huggingface-cli login
```

### 評価の設定(llm-leaderboard/configs/config.yaml の編集)
```yaml
# TODO: 松尾研のプロジェクト名、チーム名設定
# run_nameの編集
wandb:
  log: True
  entity: "weblab_lecture" # TODO 
  project: "weblab-llm-leaderboard" # TODO
  run_name: 'team1/gpt_0.125B_global_step35000_openassistant' # ご自身のteam名の後(/以降)、実験管理用にお好きな名前をつけてください

# pretrained_model_name_or_pathの編集
model:
  use_wandb_artifacts: false
  artifacts_path: ""
  pretrained_model_name_or_path: 'hotsuyuki/gpt_0.125B_global_step35000_openassistant' # huggingfaceのupload先を指定してください
  trust_remote_code: true
  device_map: "auto"
  load_in_8bit: false
  load_in_4bit: false

# pretrained_model_name_or_pathの編集
tokenizer:
  use_wandb_artifacts: false
  artifacts_path: ""
  pretrained_model_name_or_path: 'hotsuyuki/gpt_0.125B_global_step35000_openassistant' # huggingfaceのupload先を指定してください
  use_fast: true

# basemodel_name
metainfo:
  basemodel_name: "hotsuyuki/gpt_0.125B_global_step35000_openassistant" # huggingfaceのupload先を指定してください
  model_type: "open llm" # {open llm, commercial api}
  instruction_tuning_method: "None" # {"None", "Full", "LoRA", ...}
  instruction_tuning_data: ["None"] # {"None", "jaster", "dolly_ja", "oasst_ja", ...}
  num_few_shots: 0
  llm-jp-eval-version: "1.1.0"

# for llm-jp-eval
max_seq_length: 1024 # モデルの扱えるcontext lengthに応じて調整してください

# for mtbench
mtbench:
  max_new_token: 256 # モデルの扱えるcontext lengthに応じて調整してください
```

### 評価の実行

```
cd llm-leaderboard
python scripts/run_eval.py
```

### リーダーボードの確認
https://wandb.ai/weblab_lecture/weblab-llm-leaderboard

