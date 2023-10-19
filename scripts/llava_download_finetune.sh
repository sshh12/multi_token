#!/bin/bash
# https://github.com/SkunkworksAI/BakLLaVA/blob/main/setup_finetune.sh

# Create directories
mkdir -p /data/llava_finetune_data/chat
mkdir -p /data/llava_finetune_data/images/coco/train2017
mkdir -p /data/llava_finetune_data/images/gqa/images
mkdir -p /data/llava_finetune_data/images/ocr_vqa/images
mkdir -p /data/llava_finetune_data/images/textvqa/train_images
mkdir -p /data/llava_finetune_data/images/vg/VG_100K
mkdir -p /data/llava_finetune_data/images/vg/VG_100K_2

# Download datasets
wget -P /data/llava_finetune_data/chat/ https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json
wget -P /data/llava_finetune_data/images/coco/train2017/ http://images.cocodataset.org/zips/train2017.zip
wget -P /data/llava_finetune_data/images/gqa/images/ https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
wget -P /data/llava_finetune_data/images/vg/VG_100K_2/ https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
wget -P /data/llava_finetune_data/images/vg/VG_100K/ https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -P /data/llava_finetune_data/images/textvqa/train_images/ https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

# Unzip datasets
unzip /data/llava_finetune_data/images/coco/train2017/train2017.zip -d /data/llava_finetune_data/images/coco/train2017/
unzip /data/llava_finetune_data/images/gqa/images/images.zip -d /data/llava_finetune_data/images/gqa/images/
unzip /data/llava_finetune_data/images/vg/VG_100K_2/images2.zip -d /data/llava_finetune_data/images/vg/VG_100K_2/
unzip /data/llava_finetune_data/images/vg/VG_100K/images.zip -d /data/llava_finetune_data/images/vg/VG_100K/
unzip /data/llava_finetune_data/images/textvqa/train_images/train_val_images.zip -d /data/llava_finetune_data/images/textvqa/train_images/

# Remove zip files
rm /data/llava_finetune_data/images/coco/train2017/train2017.zip
rm /data/llava_finetune_data/images/gqa/images/images.zip
rm /data/llava_finetune_data/images/vg/VG_100K_2/images2.zip
rm /data/llava_finetune_data/images/vg/VG_100K/images.zip
rm /data/llava_finetune_data/images/textvqa/train_images/train_val_images.zip