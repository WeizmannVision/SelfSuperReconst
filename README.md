
# Self-Supervised RGBD Reconstruction from Brain Activity 🧠

***Official PyTorch implementation & pretrained models for:***
> **More than meets the eye: Self-supervised depth reconstruction from brain activity** \
> *Guy Gaziv, Michal Irani*  [[`arXiv`](https://arxiv.org/abs/2106.05113)]

> **Self-Supervised Natural Image Reconstruction and Rich Semantic Classification from Brain Activity** \
> *Guy Gaziv\*, Roman Beliy\*, Niv Granot\*, Assaf Hoogi, Francesca Strappini, Tal Golan, Michal Irani*  \
> [[`Project Page`](http://www.wisdom.weizmann.ac.il/~vision/SSReconstnClass/) | [`arXiv`](https://arxiv.org/abs/2106.05113) | [`Summary Video`](https://video.tau.ac.il/events/index.php?option=com_k2&view=item&id=10112:fmri&Itemid=550)]

<div align="center">
  <img width="100%" alt="Summary" src=".github/summary.gif">
</div>

##
### Setup
#### Code & environment
Clone this repo and create a designated conda environment using the `env.yml` file:
```
conda env create --name <envname> --file=env.yml
conda activate <envname>
```

#### Additional repo dependencies
This repo uses two additional repos: (i) [MiDaS Monocular Depth Estimation](https://github.com/isl-org/MiDaS/releases/tag/v1), and (ii) [Perceptual Similarity Metric](https://github.com/richzhang/PerceptualSimilarity).
Clone these repos and place under the parent folder of this repo. Imports to code from there are made accordingly.
You can dismiss this step depending on your needs: (i) MiDaS is dynamically used to estimate depth from RGB data on-the-fly either as part of inference or reconstruction criterion (`midas_loss`) -- but not necessarily, depending on your decoder training configuration. (ii) lpips is used to evaluate RGB reconstructions.
If any of these is not required just comment out imports/relevant code parts.

####  Data
This code requires all necessary data to be placed/linked under `data` folder in the following structure. *For completeness and ease of demo only*, we provide these for **download from [HERE](https://github.com/WeizmannVision/SelfSuperReconst/releases).** \
**Please refer to the original datasets behind these derivatives alongside their proper citation ([fMRI on ImageNet](https://openneuro.org/datasets/ds001246/versions/1.0.1), [ILSVRC](https://image-net.org/challenges/LSVRC/index.php)).**
```
/data
┣ 📂 imagenet
┃	┣ 📂 val 
┃ 	┃	┗ (ImageNet validation images by original class folders)

┣ 📂 imagenet_depth
┃	┣ 📂 val_depth_on_orig_small_png_uint8
┃	┃	┗ (depth component of ImageNet validation images using MiDaS small model)
┃	┣ 📂 val_depth_on_orig_large_png_uint8
┃	┃	┗ (depth component of ImageNet validation images using MiDaS large model)

┣ 📂 imagenet_rgbd
┃	┗	(pretrained depth-only & RGBD vgg16/19 model checkpoints optimized for ImageNet classification challenge; These are used as Encoder backbone net or as a reconstruction metric)

┣ 📜 images_112.npz (fMRI on ImageNet stimuli at resolution 112x112)
┣ 📜 rgbd_112_from_224_large_png_uint8.npz (saved RGBD data at resolution 112, depth computed on 224 stimuli using MiDaS large model and saved as PNG uint8)
┣ 📜 sbj_<X>.npz (fMRI data)
┗ 📜 model-<X>.pt (MiDaS depth estimation models)
```
In addition, the original MiDaS depth estimation models ([model-f6b98070.pt](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt)
and [model-small-70d6b9c8.pt](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt)) should be downloaded from the [MiDaS original repo](https://github.com/isl-org/MiDaS/releases/tag/v1) and placed under `data`.

##
### Training
The `scripts` folder provides most of the basic utility and experiments. In a nutshell, the training is comprised of two phases: (i) Encoder training implemented in `train_encoder.py`, followed by (ii) Decoder training, implemented in `train_decoder.py`. 
Each of those scripts need be run with the relevant flags which are listed in config files. General flags for both Encoder & Decoder training are listed in `config.py`, and Encoder/Decoder-training specific flags in `config_enc.py` or `config_dec.py`, respectively.

#### Example 1 (RGB-only):
Train RGB-only Encoder (supervised-only):
```
python $(scripts/train_enc_rgb.sh)
```
Then train RGB-only Decoder (supervised + self-supervised):
```
python $(scripts/train_dec_rgb.sh)
```
The results (reconstructions of train and test images) will appear under `results`. Detailed tensorboard logs will output under `<tensorboard_log_dir>` (to be set within the scripts).

#### Example 2 (Depth-only):

`python $(scripts/train_enc_depth.sh)` followed by `python $(scripts/train_dec_depth.sh)`

#### Example 3 (RGBD):
`python $(scripts/train_enc_rgbd.sh)` followed by `python $(scripts/train_dec_rgbd.sh)`

##
### Evaluation
The `eval.ipynb` notebook provides functionality for evaluating reconstruction quality via n-way identification experiments (two types: % correct or rank identification, see paper).
The DataFrame with evaluation results is saved under `eval_results` folder as a `.pkl` file. 
The `eval_plot.ipynb` loads these data and implements some basic visualization and printing of results.

##
### Citation
If you find this repository useful, please consider giving a star ⭐️ and citation:
```
@article{Gaziv2021MoreActivity, 
	title = {{More than meets the eye: Self-supervised depth reconstruction from brain activity}}, 
	author = {Gaziv, Guy and Irani, Michal}, 
	journal={arXiv preprint arXiv:2106.05113},	
	year = {2021}
}

@article{Gaziv2020, 
	title = {{Self-Supervised Natural Image Reconstruction and Rich Semantic Classification from Brain Activity}}, 
	author = {Gaziv, Guy and Beliy, Roman and Granot, Niv and Hoogi, Assaf and Strappini, Francesca and Golan, Tal and Irani, Michal}, 
	journal = {bioRxiv},
	year = {2020}
}
```
