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
※※※※※注意※※※※※  
モデル名, run_name, max_seq_length, max_new_token以外は編集しないでください

```yaml
# run_nameの編集
wandb:
  log: True
  entity: "weblab-geniac-leaderboard"
  project: "leaderboard"
  run_name: 'weblab-geniacN/gpt_0.125B_global_step35000_openassistant' # ご自身のteam名(weblab-geniac{N}, Nは1~8)の後(/以降)、実験管理用にお好きな名前をつけてください

# pretrained_model_name_or_pathの編集
model:
  pretrained_model_name_or_path: 'hotsuyuki/gpt_0.125B_global_step35000_openassistant' # huggingfaceのupload先を指定してください

# pretrained_model_name_or_pathの編集
tokenizer:
  pretrained_model_name_or_path: 'hotsuyuki/gpt_0.125B_global_step35000_openassistant' # huggingfaceのupload先を指定してください

# basemodel_name
metainfo:
  basemodel_name: "hotsuyuki/gpt_0.125B_global_step35000_openassistant" # huggingfaceのupload先を指定してください

# for llm-jp-eval
max_seq_length: 1024 # モデルの扱えるcontext lengthに応じて調整してください

# for mtbench
mtbench:
  max_new_token: 256 # モデルの扱えるcontext lengthに応じて調整してください
```

### 評価の実行

```
cd llm-leaderboard
# llm-jp-eval, japanese-mt-bench(API利用料がかかる)両方動かす
python scripts/run_eval.py
# llm-jp-evalのみ動かす
python scripts/run_llmjp_eval.py
# japanese-mt-bench(API利用料がかかる)のみ動かす
python scripts/run_jmtbench_eval.py
```

### リーダーボードの確認
https://wandb.ai/weblab-geniac-leaderboard/leaderboard/reports/-Geniac---Vmlldzo3Mjg0MTY4