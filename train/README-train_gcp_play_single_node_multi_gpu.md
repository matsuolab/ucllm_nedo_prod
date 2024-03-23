# Train

## 前提

* 計算環境: g2, 1 node, 1 GPU (Nvidia L4 24GB)
  * 例: `$ srun --partition g2 --nodes=1 --gpus-per-node=1 --time=04:00:00 -c 12 --pty bash -i`

## Step 0. 環境構築

このステップでの目標は、下記のようなディレクトリ構造の状態になることです。

Before:
```sh
~/ucllm_nedo_dev/
└── train/
    ├── scripts/
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

After:
```sh
~/ucllm_nedo_dev/
└── train/
    ├── .venv/
    ├── apex/
    ├── llm-jp-sft/
    ├── Megatron-DeepSpeed/
    ├── scripts/
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

### Step 0-1. Python仮想環境作成前における下準備

```sh
$ cd ~/

# condaのインストール先ディレクトリを作成。
$ mkdir -p ~/miniconda3/ && cd ~/miniconda3/

# condaをインストール。
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh && bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -b -u -p ~/miniconda3/

# インストールしたcondaを有効化。
$ source ~/miniconda3/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
$ which conda && echo "====" && conda --version
```

### Step 0-2. Python仮想環境の作成

```sh
$ cd ~/ucllm_nedo_dev/train/

# Python仮想環境を作成。
$ conda create --name .venv python=3.9 -y

# Python仮想環境を有効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を編集するように設定。
$ mkdir -p ~/miniconda3/envs/.venv/etc/conda/activate.d
$ echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > ~/miniconda3/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
$ echo 'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/.venv/lib:$LD_LIBRARY_PATH"' >> ~/miniconda3/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
$ chmod +x ~/miniconda3/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh

# Python仮想環境を無効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を元に戻すように設定。
$ mkdir -p ~/miniconda3/envs/.venv/etc/conda/deactivate.d
$ echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > ~/miniconda3/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
$ echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> ~/miniconda3/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
$ chmod +x ~/miniconda3/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh

# 作成したPython仮想環境を有効化。
$ conda activate .venv

# cuda-11.8.0をインストール。
$ conda install nvidia/label/cuda-11.8.0::cuda-toolkit

# PyTorchを指定のバージョンでインストール。
$ conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Python仮想環境を有効化した後は (python3コマンドだけでなく) pythonコマンドも使えることを確認。
(.venv) $ which python && echo "====" && python --version

# 環境変数 `$PATH` に `$HOME/miniconda3/envs/.venv/bin` が含まれていることを確認。
(.venv) $ echo $PATH

# 環境変数 `$LD_LIBRARY_PATH` に `$HOME/miniconda3/envs/.venv/lib` が含まれていることを確認。
(.venv) $ echo $LD_LIBRARY_PATH
```

### Step 0-3. パッケージ等のインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# PyTorchを指定のバージョンでインストールした後に、requirements.txtを用いて諸々のパッケージをインストール。
(.venv) $ pip install -r ~/ucllm_nedo_dev/train/requirements.txt

# deepspeedの依存パッケージをインストール。
(.venv) $ pip install deepspeed-kernels

# deepspeedを指定のバージョンでインストール。このとき、deepspeed関連の拡張機能たち "ops" を事前にビルドしておくために `DS_BUILD_OPS=1` と設定。 
# https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops
# ※しばらく時間がかかるので注意。
(.venv) $ DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.12.4

# deepspeed関連の拡張機能たち "ops" が正しくインストールされていることを確認。
(.venv) $ ds_report
```

### Step 0-4. Megatron-DeepSpeedのインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# Megatron-DeepSpeedのレポジトリをクローン。
(.venv) $ git clone https://github.com/hotsuyuki/Megatron-DeepSpeed

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/ && git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240205.1.0

# Megatron-DeepSpeedをインストール。
(.venv) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/ && python setup.py install
```

### Step 0-5. apexのインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# apexのレポジトリをクローン。
(.venv) $ git clone https://github.com/NVIDIA/apex

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv) $ cd ~/ucllm_nedo_dev/train/apex/ && git fetch origin && git checkout refs/tags/23.08

# nvccが対応しているCUDAのバージョンとPyTorchが依存しているCUDAのバージョンが一致していることを確認。
(.venv) $ which nvcc && echo "====" && nvcc --version && echo "====" && python -c "import torch; print(torch.__version__)"

# pipのバージョンが23.1以上であることを確認。
(.venv) $ which pip && echo "====" && pip --version

# pipのバージョンが23.1以上の場合のインストール方法で、apexをインストール。
# ※しばらく時間がかかるので注意。
(.venv) $ cd ~/ucllm_nedo_dev/train/apex/ && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# apexがインストールされていることを確認。
(.venv) $ pip list | grep "apex"

# apex_C.cpython-311-x86_64-linux-gnu.soが作成されていることを確認。
(.venv) $ find ~/ucllm_nedo_dev/train/apex/build/lib.linux-x86_64-cpython-311/ -name apex_C.cpython-311-x86_64-linux-gnu.so
```

### Step 0-6. Flash Attention 2のインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# Flash Attention 2のインストールに必要なninjaを念のため再インストール。
(.venv) $ pip uninstall ninja -y && pip install ninja==1.11.1

# Flash Attention 2をインストール。
(.venv) $ pip install flash-attn==2.5.0 --no-build-isolation

# Flash Attention 2がインストールされていることを確認。
(.venv) $ pip list | grep "flash-attn"
```

### Step 0-7. llm-jp-sftのインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# llm-jp-sftのレポジトリをクローン。
(.venv) $ git clone https://github.com/hotsuyuki/llm-jp-sft

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv) $ cd ~/ucllm_nedo_dev/train/llm-jp-sft/ && git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240208.1.0
```

## Step 1. トークナイザーの学習

### Step 1-1. 学習の実行

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step1_train_tokenizer/

# 学習スクリプトを実行。
(.venv) $ python ./train_sentencepiece_tokenizer.py \
    --input ./dataset/botchan.txt \
    --model_prefix botchan \
    --vocab_size 2000

# 出力された学習済みトークナイザーを出力ディレクトリへ移動。
(.venv) $ mkdir -p ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/ && mv ./botchan.model ./botchan.vocab --target-directory ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/
```

## Step 2. モデルの事前学習

### Step 2-1. 事前学習の実行

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step2_pretrain_model/

# W&Bにログイン。
# https://wandb.ai/settings --> Danger Zone --> API keys --> APIキーをコピペ。
(.venv) $ wandb login

# W&Bにログインしていることを確認。
(.venv) $ cat ~/.netrc

# 事前学習スクリプトを実行。
(.venv) $ bash ./gcp_node-1_gpu/dataset-arxiv_tokenizer-sentencepiece_model-gpt_0.125B/zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
    --input_tokenizer_file ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/botchan.model \
    --output_model_dir ~/ucllm_nedo_dev/train/output/step2_pretrain_model/ \
    --save_interval 1000
```

### Step 2. でのトラブルシューティング

##### 1. "ImportError: cannot import name 'helpers' from 'megatron.data' (Megatron-DeepSpeed/megatron/data/__init__.py)" というエラーが出た場合

原因: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/helpers.cpython-311-x86_64-linux-gnu.so` が正しく作成されていないことが原因と考えられます。

解決策: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefile` 内に記載されている `python3-config` のパスを `$ which python3-config` で出力された絶対パスに変更してから、 `~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/` にて `make` コマンドを実行してみて下さい。

```sh
# python3-configの絶対パスを確認。
(.venv) $ which python3-config

# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefileのpython3-configのパスを、上記のwhichコマンドで出力された絶対パスに変更。
(.venv) $ vim ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefile
"""
# Before
LIBEXT = $(shell python3-config --extension-suffix)

# After
LIBEXT = $(shell /absolute/path/to/python3-config --extension-suffix)
"""

# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/にてmakeコマンドを実行。
(.venv) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/ && make

# helpers.cpython-311-x86_64-linux-gnu.soが作成されていることを確認。
(.venv) $ find ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/ -name helpers.cpython-311-x86_64-linux-gnu.so
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
(.venv) $ rm -rf ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/
```

参考リンク: <br/>
* https://github.com/NVIDIA/Megatron-LM/issues/82#issuecomment-1613749424

## Step 3. 事前学習済みモデルのアップロード

### Step 3-1. トークナイザーと事前学習済みモデルのHuggingFace Transformers形式への変換

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/

# 変換スクリプトを実行。
(.venv) $ bash ./convert_tokenizer_and_pretrained_model_to_huggingface_transformers.sh \
    --input_tokenizer_file ~/ucllm_nedo_dev/train/output/step1_train_tokenizer/botchan/botchan.model \
    --input_model_dir ~/ucllm_nedo_dev/train/output/step2_pretrain_model/checkpoint/gpt_0.125B_${YOUR_JOBNAME}/global_step1000/ \
    --output_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step3_upload_pretrained_model/gpt_0.125B_global_step1000/
```

### Step 3-2. トークナイザーと事前学習済みモデルのHuggingFace Hubへのアップロード

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/

# HuggingFaceにログイン。
# https://huggingface.co/settings/tokens --> 書き込み権限ありのAPIキーをコピペ。
(.venv) $ huggingface-cli login

# HuggingFaceにログインしていることを確認。
(.venv) $ huggingface-cli whoami

# アップロードスクリプトを実行。
(.venv) $ python ./upload_tokenizer_and_pretrained_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step3_upload_pretrained_model/gpt_0.125B_global_step1000/ \
    --output_model_name gpt_0.125B_global_step1000 \
    --test_prompt_text "Once upon a time,"
```

## Step 4. モデルのファインチューニング

### Step 4-1. ファインチューニングの実行

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step4_finetune_model/

# ファインチューニングスクリプトを実行。 (HuggingFaceにアップロードした事前学習モデルをダウンロードして使用する場合)
(.venv) $ bash ./gcp_play_node-1_gpu/dataset-openassistant_tokenizer-sentencepiece_model-gpt_0.125B/launcher-none_zero-none.sh --input_model_name_or_path ${YOUR_HUGGINGFACE_USERNAME}/gpt_0.125B_global_step1000 \
    --output_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/

# ファインチューニングスクリプトを実行。 (ローカルに保存してある事前学習モデルをそのまま使用する場合)
(.venv) $ bash ./gcp_play_node-1_gpu/dataset-openassistant_tokenizer-sentencepiece_model-gpt_0.125B/launcher-none_zero-none.sh --input_model_name_or_path ~/ucllm_nedo_dev/train/output/step3_upload_pretrained_model/gpt_0.125B_global_step1000/ \
    --output_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/
```

## Step 5. ファインチューニング済みモデルのアップロード

### Step 5-1. トークナイザーとファインチューニング済みモデルのHuggingFace Hubへのアップロード

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step5_upload_finetuned_model/

# HuggingFaceにログインしていることを確認。
(.venv) $ huggingface-cli whoami

# アップロードスクリプトを実行。
(.venv) $ python ./upload_tokenizer_and_finetuned_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/ \
    --output_model_name gpt_0.125B_global_step1000_openassistant \
    --test_prompt_text "Once upon a time,"
```
