#!/bin/bash
# https://github.com/SkunkworksAI/BakLLaVA/blob/main/setup_pretrain.sh

# Create directories
mkdir -p /data/llava_pretrain_data/chat
mkdir -p /data/llava_pretrain_data/images

# Download datasets
wget -P /data/llava_pretrain_data/chat/ https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
wget -P /data/llava_pretrain_data/images/ https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip

# Unzip datasets
unzip /data/llava_pretrain_data/images/images.zip -d /data/llava_pretrain_data/images/

# Remove zip files
rm /data/llava_pretrain_data/images/images.zip