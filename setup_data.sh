#!/bin/bash
# Author: Guy Gaziv

set -e

trap "exit 1" TERM
export TOP_PID=$$

if [ -d data/ ]; then
    rm -rf data/
fi

mkdir data
cd data

RELEASE_PATH=https://github.com/WeizmannVision/SelfSuperReconst/releases/download/v1/

# 'fMRI on ImageNet' presented images at resolution of 112
wget $RELEASE_PATH/images_112.npz

# 'fMRI on ImageNet' presented images at resolution of 112 with their depth channel, 
# which was computed on 224x224 images using MiDaS large model and saved as PNG uint8
wget $RELEASE_PATH/rgbd_112_from_224_large_png_uint8.npz

# 'fMRI on ImageNet' subject preprocessed fMRI data
for sbj_num in 1 2 3 4 5
do
    wget $RELEASE_PATH/sbj_${sbj_num}.npz        
done

# MiDaS pretrained models for depth estimation from images
wget https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt
wget https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt

# ImageNet validation data
mkdir imagenet
cd imagenet
filename=imagenet_val.tar.gz
wget $RELEASE_PATH/$filename
tar xvzf $filename
rm -f $filename
cd ..

# Depth estimated over ImageNet validation data using both small/large MiDaS model
mkdir imagenet_depth
cd imagenet_depth
filename=val_depth_on_orig_small_png_uint8.tar.gz
wget $RELEASE_PATH/$filename
tar xvzf $filename
rm -f $filename

filename=val_depth_on_orig_large_png_uint8.tar.gz
wget $RELEASE_PATH/$filename
tar xvzf $filename
rm -f $filename
cd ..

# Pretrained checkpoints optimized for ILSVRC from depth input or RGBD
mkdir imagenet_rgbd
cd imagenet_rgbd
wget $RELEASE_PATH/vgg16_depth_only_large_norm_within_img_best.pth.tar
wget $RELEASE_PATH/vgg16_depth_only_norm_within_img_best.pth.tar
wget $RELEASE_PATH/vgg16_rgbd_large_norm_within_img_best.pth.tar
wget $RELEASE_PATH/vgg19_depth_only_large_norm_within_img_best.pth.tar
wget $RELEASE_PATH/vgg19_rgbd_large_norm_within_img_best.pth.tar

cd ..