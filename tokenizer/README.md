# Tokenizer
ucllm-nedo-dev配下に配置されることを想定しています。

# GCPでの実行手順
```bash
#実行環境
$ srun --gpus-per-node=0 --time=06:00:00 --nodes=1 --pty bash -i
#condaを有効化。
$ source ~/miniconda3/etc/profile.d/conda.sh
# Python仮想環境を有効化。
$ conda activate .venv_data

# トークナイザーの作業ディレクトリへ移動
$ cd ~/ucllm_nedo_dev/tokenizer
```

## 1. Download datasets

### [日本語wikipedia dump](https://dumps.wikimedia.org/jawiki/)
```bash
$ python -m preprocessing.download_dataset --split=20240301 --language=ja --output_base=/persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer
```
### [英語wikipedia dump](https://dumps.wikimedia.org/enwiki/)
```bash
$ python -m preprocessing.download_dataset --split=20240301 --language=en --output_base=/persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer
```

## 2. データ整形

### 事前準備
```bash
# Python仮想環境を有効化。
$ conda deactivate
# Python仮想環境を有効化。（wikiextractorはpython3.10でしか動かない）
$ conda activate .venv
$ pip install wikiextractor
```
### 日本語
```bash
$ python -m wikiextractor.WikiExtractor -o /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/prefilter/ja/ --no-templates /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/tmp/wikipedia/20240301/ja/jawiki-20240301-pages-articles-multistream.xml.bz2
```
### 英語
```bash
$ python -m wikiextractor.WikiExtractor -o /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/prefilter/en/ --no-templates /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/tmp/wikipedia/20240301/en/enwiki-20240301-pages-articles-multistream.xml.bz2
```

## 3. jsonl作成（英語は乱択）

### 事前準備
```bash
# Python仮想環境を有効化。
$ conda deactivate
# Python仮想環境を有効化。
$ conda activate .venv_data
```
### 日本語
```bash
$ python -m preprocessing.t01_delete_spaceline \
    --language ja \
    --input_base /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/prefilter/ \
    --output_base /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/jsonl/
```
### 英語
```bash
$ python -m preprocessing.t01_delete_spaceline \
    --language en \
    --input_base /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/prefilter/ \
    --output_base /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/jsonl/
```

## 4. cleaning and text作成

```bash
$ python -m preprocessing.filtering \
    --input_dir /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/jsonl/ \
    --output_dir /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/filter/
```

## 5. 未知語の間引き（日本語のみ）

### 事前準備
```bash
pip install fugashi[unidic]
python -m  unidic download
```
### 日本語
```bash
$ python -m preprocessing.t02_mabiki \
    --input /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/filter/ja_wiki/filtering.txt \
    --output /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/text/ja_wiki_mabiki.txt
```

## 6. 分かち書き（日本語のみ）

### 日本語
```bash
$ python -m preprocessing.t03_wakachi \
    --input /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/text/ja_wiki_mabiki.txt \
    --output /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/text/jawiki_newline_mecab.txt
```

## 7. 言語ごとのトークナイズ

### 事前準備
```bash
# Python仮想環境を有効化。
$ conda deactivate
# Python仮想環境を有効化。
$ conda activate .venv
```
### 日本語
```bash
$ python -m train_tokenizer.train_sentencepiece_tokenizer \
    --input /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/text/jawiki_newline_mecab.txt \
    --model_prefix JINIAC_V0_9_ja60000 \
    --vocab_size 60000 \
    --num_threads 24 \
    --pretokenization_delimiter "||||"
```
### 英語
```bash
$ python -m train_tokenizer.train_sentencepiece_tokenizer \
    --input /persistentshare/storage/team_nakamura/member/horie/dataset/tokenizer/filter/en_wiki/filtering.txt \
    --model_prefix JINIAC_V0_9_en13000 \
    --vocab_size 13000 \
    --num_threads 24 \
    --max_sentencepiece_length 16
```

## 8. .vocabの書き換え

### 日本語
```bash
$ python train_tokenizer/replaceVocab_ja.py
```
### 英語
```bash
$ python train_tokenizer/replaceVocab_en.py
```
## 9. prefixと重複の削除

### 日本語
```bash
$ python train_tokenizer/specialSymbolRemove.py \
    JINIAC_V0_9_ja60000.vocab > JINIAC_V0_9_ja60000.vocab.symbolRemoved
```
### 英語
```bash
$ python train_tokenizer/specialSymbolRemove4symbols.py \
    JINIAC_V0_9_en13000.vocab > JINIAC_V0_9_en13000.vocab.symbolRemoved
```

## 10. 日英データマージ

### 処理
```bash
$ python train_tokenizer/mergeVocab.py \
    llm-jp-tokenizer/models/ver2.1/specialTokens.vocab \
    JINIAC_V0_9_ja60000.vocab.symbolRemoved \
    JINIAC_V0_9_en13000.vocab.symbolRemoved > JINIAC_V0_9_ja42K_en13K.merged.vocab
```

## 11. vocabファイルをmodelファイルに変換

### 処理
```bash
$ python vocab2model.py \
    --vocab JINIAC_V0_9_ja42K_en13K.merged.vocab \
    --output JINIAC_V0_9_ja42K_en13K.merged.model \
    --baseModel JINIAC_V0_9_en13000.model
```

# huggingface アップロード

## 1. トークナイザーのHuggingFace Transformers形式への変換

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/

# 変換スクリプトを実行。
(.venv_train) $ python -m convert_tokenizer_from_sentencepiece_to_huggingface_transformers \
    --input_tokenizer_file ~/ucllm_nedo_dev/tokenizer/JINIAC_V0_9_ja42K_en13K.merged.model \
    --output_tokenizer_dir ~/ucllm_nedo_dev/tokenizer/output/
```
### Step 3-2-1. トークナイザーのHuggingFace Hubへのアップロード

```sh
(.venv_train) $ cd ~/ucllm_nedo_dev/train/scripts/common/

# HuggingFaceにログイン。
# https://huggingface.co/settings/tokens --> 書き込み権限ありのAPIキーをコピペ。
(.venv_train) $ huggingface-cli login

# HuggingFaceにログインしていることを確認。
(.venv_train) $ huggingface-cli whoami

# アップロードスクリプトを実行。
(.venv_train) $ python upload_only_tokenizer_to_huggingface_hub.py \
    --input_tokenizer_dir ~/ucllm_nedo_dev/tokenizer/output \
    --output_model_name JINIAC_tokenizer_v0.9_ja42k_en13k
```