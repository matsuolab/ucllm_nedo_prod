#!/bin/bash

listings_file=$1
dataset_root=$2

BASE_URL="https://data.together.xyz/redpajama-data-v2/v1.0.0"


# download documents
while read line; do
  url="${BASE_URL}/documents/${line}.json.gz"
  dest="documents/${line}.json.gz"
  mkdir -p ${dataset_root}/$(dirname $dest)
  wget "$url" -O "${dataset_root}/$dest"
done <"$listings_file"
