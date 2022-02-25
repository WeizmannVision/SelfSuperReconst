""" Test training functionality for three types of Decoder: RGB-only, RGBD, and Depth-only.
    Use fMRI data from Subject 3 of "fMRI on ImageNet".
    Assumes GPU availability -- please set CUDA_VISIBLE_DEVICES before running.
    Assumes availability of pretrained Encoders' checkpoints.
"""
import os; GPU = os.environ['CUDA_VISIBLE_DEVICES']
import pytest
from self_super_reconst import train_decoder
from tempfile import TemporaryDirectory, NamedTemporaryFile


def get_flags(sbj_num, n_epochs, logs_dir, checkpoint_out, is_rgbd):
    if is_rgbd == 0:
        enc_cpt_name = f'sub{sbj_num}_rgb_only_best'
    elif is_rgbd == 1:
        enc_cpt_name = f'sub{sbj_num}_rgbd_best'
    elif is_rgbd == 2:
        enc_cpt_name = f'sub{sbj_num}_depth_only_best'
    return (f'program_X --exp_prefix tmp '
            f'--enc_cpt_name {enc_cpt_name} --separable 1 '
            f'--test_avg_qntl 1 --learning_rate 5e-3 --loss_weights 1,1,1 '
            '--fc_gl 1 --gl_l1 40 --gl_gl 400 --fc_mom2 0 --l1_convs 1e-4 '
            f'--tv_reg 3e-1 --n_epochs {n_epochs} --batch_size_list 24,16,48,50 '
            '--scheduler 12345 --mslr 100,140 --sched_gamma 0.2 '
            f'--percept_w 10,10,10,10,2 --rgb_mae 1 --is_rgbd {is_rgbd} '
            f'--norm_within_img 1 --sbj_num {sbj_num} --depth_from_rgb 0 '
            f'--checkpoint_out {checkpoint_out} --may_save 1 '
            f'--tensorboard_log_dir {logs_dir} --gpu {GPU}'.split())


@pytest.mark.parametrize(['sbj_num', 'n_epochs', 'is_rgbd'], [[3, 2, 0],
                                                              [3, 2, 1],
                                                              [3, 2, 2]])
def test_decoder_training_smoketest(sbj_num, n_epochs, is_rgbd):
    with TemporaryDirectory() as logs_dir, NamedTemporaryFile() as checkpoint_out_file:
        # Set flags
        train_decoder.FLAGS(get_flags(sbj_num, n_epochs, logs_dir, checkpoint_out_file.name, is_rgbd))
        # Act
        train_decoder.main([])