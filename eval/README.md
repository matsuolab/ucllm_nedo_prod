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
編集可能な項目は以下です  
- 使用するモデル  
  - pretrained_model_name_or_path  
- tokenizerの設定  
  - use_fastの指定等  
- run_name(wandbに登録するスコアのrunの名前)  
- 各タスクのcontext長の設定  
  - max_seq_length, max_new_token  
- custom_prompt_template
  - 事前学習・事後学習で使用したspecial tokenの追加は認めます、関係ない文字列(デフォルトの設定に含まれない特殊なシステムメッセージ等)の追加は認めません
  - 認める例: `<s> [INST] 以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:\n[/INST]`  
    - custom_prompt_templateを指定しなかった場合のデフォルトの設定: https://github.com/llm-jp/llm-jp-eval/blob/bbc03c655a93b244b6951f9549aad7dbf523508a/src/llm_jp_eval/utils.py#L115C28-L118
    - templateを修正する際は"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"を必ず含めるようにしてください。
  - 認めない例: `<s> [INST] <<SYS>>\n あなたは誠実で優秀な日本人のアシスタントです。 \n<</SYS>>\n\n {instruction} \n\n {input} [/INST]`
- mtbenchのtemplate(事前学習・事後学習で使用したspecial tokenの追加は認めます、関係ない文字列(デフォルトの設定に含まれない特殊なシステムメッセージ等)の追加は認めません)
  - conv_system_message
  - conv_roles
  - conv_sep
  - conv_stop_token_ids
  - conv_stop_str
  - conv_role_message_separator
  - conv_role_only_separator

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