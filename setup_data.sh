#!/bin/bash

set -e

trap "exit 1" TERM
export TOP_PID=$$

if [ -d data/ ]; then
    rm -rf data/
fi

mkdir data
cd data

RELEASE_PATH=https://github.com/WeizmannVision/SelfSuperReconst/releases/download/v1/

wget $RELEASE_PATH/images_112.npz
wget $RELEASE_PATH/rgbd_112_from_224_large_png_uint8.npz
wget -m -nd -A "sbj_*.npz" $RELEASE_PATH 
# for sbj_num in 1 2 3 4 5
# do
#     wget $RELEASE_PATH/sbj_${sbj_num}.npz        
# done

wget https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt
wget https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt

wget $RELEASE_PATH/vgg19_depth_only_large_norm_within_img_best.pth.tar
wget $RELEASE_PATH/vgg19_depth_only_norm_within_img_best.pth.tar

mkdir -p imagenet/val
mkdir -p imagenet_depth/val_depth_on_orig_small_png_uint8
mkdir -p imagenet_depth/val_depth_on_orig_large_png_uint8
mkdir imagenet_rgbd

filename=imagenet_val.tar.gz
cd imagenet/val
wget $RELEASE_PATH/$filename
tar xvzf $filename

cd ../imagenet_depth/val_depth_on_orig_small_png_uint8
filename=val_depth_on_orig_small_png_uint8.tar.gz
wget $RELEASE_PATH/$filename
tar xvzf $filename

cd ../val_depth_on_orig_large_png_uint8
filename=val_depth_on_orig_large_png_uint8.tar.gz
wget $RELEASE_PATH/$filename
tar xvzf $filename

cd ../imagenet_rgbd
wget -r -A .pth.tar $RELEASE_PATH
# wget $RELEASE_PATH/vgg16_depth_only_large_norm_within_img_best.pth.tar
# wget $RELEASE_PATH/vgg16_depth_only_norm_within_img_best.pth.tar
# wget $RELEASE_PATH/vgg19_depth_only_large_norm_within_img_best.pth.tar
# wget $RELEASE_PATH/vgg19_depth_only_norm_within_img_best.pth.tar

cd ../..