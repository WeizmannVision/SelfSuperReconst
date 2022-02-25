"""
    All flags for encoder training
    Date created: 8/29/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

from self_super_reconst.config import *

flags.DEFINE_string("tensorboard_log_dir", '/mnt/tmpfs/guyga/SelfSuperReconst/enc', "Log dir.")
flags.DEFINE_enum("roi", 'VC', ['V1', 'V2', 'V3', 'V4', 'FFA', 'PPA', 'LOC', 'LVC', 'HVC', 'VC'], '')
flags.DEFINE_string("exp_prefix", 'enc_tmp', "Experiment prefix.")

flags.DEFINE_integer("num_workers_gpu", 5, "Number of workers per GPU.")
flags.DEFINE_integer("n_epochs", 80, "Number of epochs.")

flags.DEFINE_float("learning_rate", 1e-1, "The initial value for the learning rate.")
flags.DEFINE_float("mse_loss", 1., "")
flags.DEFINE_float("cos_loss", 0.1, "")

flags.DEFINE_float("decay", .002, "Weight decay.")

flags.DEFINE_float("l1_fcreg", 0., "")
flags.DEFINE_float("l1_convs", 1e-5, "")
flags.DEFINE_float("l2_fcreg", 0., "")
flags.DEFINE_float("fc_mom2", 0., "")

flags.DEFINE_float("fc_gl", 1., "")
flags.DEFINE_float("l1_chan_mix", 5e-6, "")
flags.DEFINE_float("l1_branch_mix", 5e-2, "")

flags.DEFINE_integer("batch_size", 64, "Batch size.")

flags.DEFINE_integer("scheduler", 12345, "Reduce learning rate by scheduler.")

flags.DEFINE_float("gamma", 0.5, "Scheduler gamma")
flags.DEFINE_enum("loss", 'mse', ['mse', 'l1'], '')

flags.DEFINE_float("mix_q", 0, "Quotient of mixed samples")
flags.DEFINE_integer("pw_corr_win", 0, "Window wize for piecewise correlation.")


flags.DEFINE_integer("sum_level", 4, 'Summary level. 4: -BatchOutDist | 5: Report all.')
flags.DEFINE_enum('interp_mode', 'bicubic', ['bicubic', 'bilinear', 'trilinear'], '')

flags.DEFINE_integer("random_crop_pad_percent", 10, "")
flags.DEFINE_integer("keras_pretrained", 0, 'Load weights from keras case')

flags.DEFINE_string('bbn_arch_name', 'alexnet', '')
flags.DEFINE_integer('verbose', 0, '0: -within-epoch log | 1: Report all.')

flags.DEFINE_string('init_cpt_name', '', 'Encoder checkpoint name to load')

flags.DEFINE_integer("train_bbn", 0, 'Train the backbone network')
flags.DEFINE_integer("allow_bbn_detach", 1, 'Allow bbn detach')

init_cpt_path = lambda : 'checkpoints/{}.pth.tar'.format(FLAGS.init_cpt_name)

if __name__ == '__main__':
    pass
