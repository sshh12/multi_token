#!/bin/bash

git clone https://github.com/facebookresearch/ImageBind
cd ImageBind

echo "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d" > requirements.txt
echo "timm==0.6.7" >> requirements.txt
echo "ftfy" >> requirements.txt
echo "regex" >> requirements.txt
echo "einops" >> requirements.txt
echo "fvcore" >> requirements.txt
echo "eva-decord==0.6.1" >> requirements.txt
echo "iopath" >> requirements.txt
echo "numpy>=1.19" >> requirements.txt
echo "matplotlib" >> requirements.txt
echo "types-regex" >> requirements.txt
echo "mayavi" >> requirements.txt
echo "cartopy" >> requirements.txt
pip install -r requirements.txt
pip install -e .