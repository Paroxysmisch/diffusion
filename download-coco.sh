#!/bin/bash
mkdir -p data/annotations
cd data

# 1. Download and unzip Annotations (this contains the missing .json files)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
# This creates data/annotations/captions_train2017.json among others

# 2. Download and unzip Training Images (~19GB - This may take a while!)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 3. Download and unzip Validation Images (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
