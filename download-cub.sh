#!/bin/bash
mkdir -p data/
cd data

# 1. Download and unzip Annotations (this contains the missing .json files)
curl -L -o ./cub2002011.zip\
  https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011
unzip cub2002011.zip

