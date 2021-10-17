gpu=0  # GPU ID
sbj_num=3
tensorboard_log_dir=/mnt/tmpfs/guyga/SelfSuperReconst/enc

echo train_encoder.py \
--exp_prefix sub${sbj_num}_rgbd \
--separable 1 --n_epochs 50 --learning_rate 1e-3 --cos_loss 0.3 --random_crop_pad_percent 3 --scheduler 10 --gamma 0.2 \
--fc_gl 1 --fc_mom2 10 --l1_convs 1e-4 --is_rgbd 1 --allow_bbn_detach 1 --train_bbn 0 --norm_within_img 1 --may_save 1 \
--sbj_num $sbj_num --tensorboard_log_dir $tensorboard_log_dir --gpu $gpu