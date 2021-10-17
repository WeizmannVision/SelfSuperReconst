gpu=0  # GPU ID
sbj_num=3
tensorboard_log_dir=/mnt/tmpfs/guyga/SelfSuperReconst/dec

echo train_decoder.py --exp_prefix sub${sbj_num}_depth_only_noDE \
--enc_cpt_name sub${sbj_num}_depth_only_best --separable 1 --test_avg_qntl 1 --learning_rate 5e-3 --loss_weights 1,1,0 \
--fc_gl 1 --gl_l1 40 --gl_gl 400 --fc_mom2 0 --l1_convs 1e-4 --tv_reg 3e-1 --n_epochs 150 --batch_size_list 24,16,48,50 \
--scheduler 12345 --mslr 100,140 --sched_gamma 0.2 --percept_w 10,10,10,10,2 --rgb_mae 1 --is_rgbd 2 --norm_within_img 1 \
--sbj_num $sbj_num --depth_from_rgb 0 --tensorboard_log_dir $tensorboard_log_dir --gpu $gpu
