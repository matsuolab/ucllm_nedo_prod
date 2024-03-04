# Post Training

## 環境構築

### Pythonのバージョン確認

Python>=3.11 がインストールされているものとする

確認方法は以下
```sh
$ python --version
// Python 3.11.7
```

もしPythonがインストールされていない場合は[Python.jp](https://www.python.jp/install/centos/index.html)を参考にインストールする

### 必要なライブラリのダウンロード

preprocessingディレクトリにいることを確認した上でセットアップを行なってください

```sh
$ pwd
// ~/ucllm_redo_dev/preprocessing
$ sudo apt-get install git-lfs
```

## 1. Download datasets

### [databricks dolly Japanese](https://huggingface.co/datasets/taka-yayoi/databricks-dolly-15k-ja)


利用方法

```sh
$ python -m download_dataset --output_base=output
```

https://huggingface.co/datasets/taka-yayoi/databricks-dolly-15k-ja にアップロードされているJSONL形式のデータセットです
Dollyのトレーニングで利用可能な形式になっているので学習に応じて必要な形に整形して利用してください

#### 参考

https://www.databricks.com/jp/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm
