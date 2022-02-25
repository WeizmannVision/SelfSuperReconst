""" Test training functionality for three types of Encoder: RGB-only, RGBD, and Depth-only.
    Use fMRI data from Subject 3 of "fMRI on ImageNet".
    Assumes GPU availability -- please set CUDA_VISIBLE_DEVICES before running.
"""
import os; GPU = os.environ['CUDA_VISIBLE_DEVICES']
import pytest
from self_super_reconst import train_encoder
from tempfile import TemporaryDirectory, NamedTemporaryFile


def get_flags(sbj_num, n_epochs, logs_dir, checkpoint_out, is_rgbd):
    return (f'program_X --exp_prefix tmp --separable 1 --n_epochs {n_epochs} '
            '--learning_rate 1e-3 --cos_loss 0.3 --random_crop_pad_percent 3 '
            '--scheduler 10 --gamma 0.2 --fc_gl 1 --fc_mom2 10 --l1_convs 1e-4 '
            f'--is_rgbd {is_rgbd} --allow_bbn_detach 1 --train_bbn 0 '
            f'--norm_within_img 1 --sbj_num {sbj_num} '
            f'--tensorboard_log_dir {logs_dir} --checkpoint_out {checkpoint_out} '
            f'--may_save 1 --gpu {GPU}'.split())


@pytest.mark.parametrize(['sbj_num', 'n_epochs', 'is_rgbd'], [[3, 2, 0],
                                                              [3, 2, 1],
                                                              [3, 2, 2]])
def test_encoder_training_smoketest(sbj_num, n_epochs, is_rgbd):
    with TemporaryDirectory() as logs_dir, NamedTemporaryFile() as checkpoint_out_file:
        # Set flags
        train_encoder.FLAGS(get_flags(sbj_num, n_epochs, logs_dir, checkpoint_out_file.name, is_rgbd))
        # Act
        train_encoder.main([])
