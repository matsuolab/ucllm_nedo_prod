# Train

## 前提

* 計算環境: ABCI, 1 node, 1 GPU (Nvidia A100 40GB)
  * 例: `$ qrsh -g ${YOUR_ABCI_GROUPNAME} -l rt_AG.small=1 -l h_rt=12:00:00`

## Step 0. 環境構築

### Step 0-0. このgitレポジトリのクローン

```sh
$ cd ~/

# このレポジトリをucllm_nedo_devという名前でクローンする。
$ git clone https://github.com/matsuolab/ucllm_nedo_prod.git ucllm_nedo_dev

# ~/ucllm_nedo_dev/train以下のファイル一覧が表示されるか確認。
$ ls ~/ucllm_nedo_dev/train/
```

### Step 0-1. Python仮想環境作成前における下準備

```sh
$ cd ~/

# 念のためSSH等が故障したときなどに備えて~/.bashrcをバックアップしておく。
$ cp ~/.bashrc ~/.bashrc.backup

# moduleコマンドの初期化。
$ source /etc/profile.d/modules.sh && module purge

# Python, CUDA等を指定のバージョンでロード。
$ module load python/3.11/3.11.2 cuda/11.8/11.8.0 hpcx/2.12

# moduleコマンドでロードしたものを確認。
$ module list

# Pythonのバージョンが3.11.2になっていることを確認。
$ which python3 && echo "====" && python3 --version

# CUDAのバージョンが11.8になっていることを確認。
$ which nvcc && echo "====" && nvcc --version
```

### Step 0-2. Python仮想環境の作成

```sh
$ cd ~/ucllm_nedo_dev/train/

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
# ※このとき、 "command not found" というエラーが出た場合はそもそも既に有効化されているPython仮想環境がなかったという意味なので、問題ない。
$ deactivate

# Python仮想環境を作成。
$ python3 -m venv ~/ucllm_nedo_dev/train/.venv_train

# 作成したPython仮想環境を有効化。
# ※無効化するときのコマンドは `$ deactivate` 。
$ source ~/ucllm_nedo_dev/train/.venv_train/bin/activate

# Python仮想環境を有効化した後は (python3コマンドだけでなく) pythonコマンドも使えることを確認。
(.venv_train) $ which python && echo "====" && python --version
```

### Step 0-3. パッケージ等のインストール

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/

# pipを指定のバージョンでインストール。
(.venv_train) $ pip install pip==24.0

# PyTorchを指定のバージョンでインストール。
(.venv_train) $ pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html

# PyTorchを指定のバージョンでインストールした後に、requirements.txtを用いて諸々のパッケージをインストール。
(.venv_train) $ pip install -r ~/ucllm_nedo_dev/train/requirements.txt

# deepspeedの依存パッケージをインストール。
(.venv_train) $ pip install deepspeed-kernels

# deepspeedを指定のバージョンでインストール。このとき、deepspeed関連の拡張機能たち "ops" を事前にビルドしておくために `DS_BUILD_OPS=1` と設定。
# https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops
# ※しばらく時間がかかるので注意。
(.venv_train) $ DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.12.4

# deepspeed関連の拡張機能たち "ops" が正しくインストールされていることを確認。
(.venv_train) $ ds_report
```

### Step 0-4. Megatron-DeepSpeedのインストール

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/

# Megatron-DeepSpeedのレポジトリをクローン。
(.venv_train) $ git clone https://github.com/hotsuyuki/Megatron-DeepSpeed.git

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv_train) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/ && git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240411.1.0

# Megatron-DeepSpeedをインストール。
(.venv_train) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/ && python setup.py install
```

### Step 0-5. apexのインストール

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/

# apexのレポジトリをクローン。
(.venv_train) $ git clone https://github.com/NVIDIA/apex.git

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv_train) $ cd ~/ucllm_nedo_dev/train/apex/ && git fetch origin && git checkout refs/tags/23.08

# nvccが対応しているCUDAのバージョンとPyTorchが依存しているCUDAのバージョンが一致していることを確認。
# https://github.com/matsuolab/ucllm_nedo_prod/pull/5 by https://github.com/awakia
(.venv_train) $ which nvcc && echo "====" && nvcc --version && echo "====" && python -c "import torch; print(f'{torch.version.cuda = }')"

# pipのバージョンが23.1以上であることを確認。
(.venv_train) $ which pip && echo "====" && pip --version

# pipのバージョンが23.1以上の場合のインストール方法で、apexをインストール。
# ※しばらく時間がかかるので注意。
(.venv_train) $ cd ~/ucllm_nedo_dev/train/apex/ && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# apexがインストールされていることを確認。
(.venv_train) $ pip list | grep "apex"

# apex_C.cpython-311-x86_64-linux-gnu.soが作成されていることを確認。
(.venv_train) $ find ~/ucllm_nedo_dev/train/apex/build/lib.linux-x86_64-cpython-311/ -name apex_C.cpython-311-x86_64-linux-gnu.so
```

### Step 0-6. Flash Attention 2のインストール

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/

# Flash Attention 2のインストールに必要なninjaを念のため再インストール。
(.venv_train) $ pip uninstall ninja -y && pip install ninja==1.11.1

# Flash Attention 2をインストール。
(.venv_train) $ pip install flash-attn==2.5.0 --no-build-isolation

# Flash Attention 2がインストールされていることを確認。
(.venv_train) $ pip list | grep "flash-attn"
```

### Step 0-7. llm-jp-sftのインストール

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/

# llm-jp-sftのレポジトリをクローン。
(.venv_train) $ git clone https://github.com/hotsuyuki/llm-jp-sft.git

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv_train) $ cd ~/ucllm_nedo_dev/train/llm-jp-sft/ && git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240407.1.0
```

## Step 1. トークナイザーの学習

### Step 1-1. 学習の実行

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/step1_train_tokenizer/

# 学習スクリプトを実行。
(.venv_train) $ python ./train_sentencepiece_tokenizer.py \
    --input ./dataset/botchan.txt \
    --model_prefix botchan \
    --vocab_size 2000

# 出力された学習済みトークナイザーを出力ディレクトリへ移動。
(.venv_train) $ mkdir -p ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/ && mv ./botchan.model ./botchan.vocab --target-directory ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/
```

## Step 2. モデルの事前学習

### Step 2-1. 事前学習の実行

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/step2_pretrain_model/

# W&Bにログイン。
# https://wandb.ai/settings --> Danger Zone --> API keys --> APIキーをコピペ。
(.venv_train) $ wandb login

# W&Bにログインしていることを確認。
(.venv_train) $ cat ~/.netrc

# 事前学習スクリプトを実行。
(.venv_train) $ bash ./abci_node-1_gpu-1/dataset-arxiv_tokenizer-sentencepiece_model-gpt_0.125B/zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
    --input_tokenizer_file ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/botchan.model \
    --output_model_dir ~/ucllm_nedo_dev/train/output/step2_pretrain_model/ \
    --save_interval 1000 \
    --wandb_entity ${YOUR_WANDB_ENTITY_OR_TEAM_NAME} \
    --wandb_project ${YOUR_WANDB_PROJECT_NAME}
```

### Step 2. でのトラブルシューティング

##### 1. "ImportError: cannot import name 'helpers' from 'megatron.data' (Megatron-DeepSpeed/megatron/data/__init__.py)" というエラーが出た場合

原因: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/helpers.cpython-311-x86_64-linux-gnu.so` が正しく作成されていないことが原因と考えられます。

解決策: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefile` 内に記載されている `python3-config` のパスを `$ which python3-config` で出力された絶対パスに変更してから、 `~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/` にて `make` コマンドを実行してみて下さい。

```sh
# python3-configの絶対パスを確認。
(.venv_train) $ which python3-config

# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefileのpython3-configのパスを、上記のwhichコマンドで出力された絶対パスに変更。
(.venv_train) $ vim ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefile
"""
# Before
LIBEXT = $(shell python3-config --extension-suffix)

# After
LIBEXT = $(shell /absolute/path/to/python3-config --extension-suffix)
"""

# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/にてmakeコマンドを実行。
(.venv_train) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/ && make

# helpers.cpython-311-x86_64-linux-gnu.soが作成されていることを確認。
(.venv_train) $ find ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/ -name helpers.cpython-311-x86_64-linux-gnu.so
```

参考リンク: <br/>
* https://zenn.dev/turing_motors/articles/04c1328bf6095a#pyenv-virtualenv-%E3%82%92%E4%BD%BF%E3%81%86%E3%81%A8%E5%BF%85%E8%A6%81%E3%81%AB%E3%81%AA%E3%82%8B%E5%87%A6%E7%90%86
* https://zenn.dev/turing_motors/articles/da7fa101ecb9a1#makefile%E3%81%AE%E6%9B%B8%E3%81%8D%E6%8F%9B%E3%81%88

#### 2. 事前学習スクリプトが "> compiling and loading fused kernels ..." というところでスタックした場合

原因: <br/>
既存の `~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/` が作成された当時と現在でハードウェアやCUDAのバージョンが異なっていることが原因と考えられます。

解決策: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/` を削除してから、もう一度事前学習スクリプトを実行してみて下さい。

```sh
# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/を削除。
(.venv_train) $ rm -rf ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/
```

参考リンク: <br/>
* https://github.com/NVIDIA/Megatron-LM/issues/82#issuecomment-1613749424

## Step 3. 事前学習済みモデルのアップロード

### Step 3-1. トークナイザーと事前学習済みモデルのHuggingFace Transformers形式への変換

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/

# 変換スクリプトを実行。
(.venv_train) $ bash ./convert_tokenizer_and_pretrained_model_to_huggingface_transformers.sh \
    --input_tokenizer_file ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/botchan.model \
    --input_model_dir ~/ucllm_nedo_dev/train/output/step2_pretrain_model/checkpoint/gpt_0.125B_${YOUR_JOBNAME}/global_step1000/ \
    --output_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step3_upload_pretrained_model/gpt_0.125B_global_step1000/
```

### Step 3-2. トークナイザーと事前学習済みモデルのHuggingFace Hubへのアップロード

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/

# HuggingFaceにログイン。
# https://huggingface.co/settings/tokens --> 書き込み権限ありのAPIキーをコピペ。
(.venv_train) $ huggingface-cli login

# HuggingFaceにログインしていることを確認。
(.venv_train) $ huggingface-cli whoami

# アップロードスクリプトを実行。
(.venv_train) $ python ./upload_tokenizer_and_pretrained_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step3_upload_pretrained_model/gpt_0.125B_global_step1000/ \
    --output_model_name gpt_0.125B_global_step1000 \
    --test_prompt_text "Once upon a time,"
```

## Step 4. モデルのファインチューニング

### Step 4-1. ファインチューニングの実行

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/step4_finetune_model/

# ファインチューニングスクリプトを実行。 (HuggingFaceにアップロードした事前学習モデルをダウンロードして使用する場合)
(.venv_train) $ bash ./abci_node-1_gpu-1/dataset-openassistant/launcher-none_zero-none.sh \
    --input_model_name_or_path ${YOUR_HUGGINGFACE_USERNAME}/gpt_0.125B_global_step1000 \
    --output_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/ \
    --wandb_entity ${YOUR_WANDB_ENTITY_OR_TEAM_NAME} \
    --wandb_project ${YOUR_WANDB_PROJECT_NAME}

# ファインチューニングスクリプトを実行。 (ローカルに保存してある事前学習モデルをそのまま使用する場合)
(.venv_train) $ bash ./abci_node-1_gpu-1/dataset-openassistant/launcher-none_zero-none.sh \
    --input_model_name_or_path ~/ucllm_nedo_dev/train/output/step3_upload_pretrained_model/gpt_0.125B_global_step1000/ \
    --output_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/ \
    --wandb_entity ${YOUR_WANDB_ENTITY_OR_TEAM_NAME} \
    --wandb_project ${YOUR_WANDB_PROJECT_NAME}
```

## Step 5. ファインチューニング済みモデルのアップロード

### Step 5-1. トークナイザーとファインチューニング済みモデルのHuggingFace Hubへのアップロード

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/step5_upload_finetuned_model/

# HuggingFaceにログインしていることを確認。
(.venv_train) $ huggingface-cli whoami

# アップロードスクリプトを実行。
(.venv_train) $ python ./upload_tokenizer_and_finetuned_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/ \
    --output_model_name gpt_0.125B_global_step1000_openassistant \
    --test_prompt_text "Once upon a time,"
```
