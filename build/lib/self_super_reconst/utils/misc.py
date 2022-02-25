"""
    Misc. functionality.
    Date created: 8/25/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

from matplotlib import pyplot as plt
from datetime import datetime
import time
from termcolor import cprint
from tqdm import tqdm
from PIL import Image
import os, numpy as np, pandas as pd, seaborn as sns, shutil, itertools, contextlib, copy, glob
from os.path import join as pjoin
import torch, types
import torch.nn as nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter as SumWr
from torchvision.utils import make_grid, save_image
from absl import flags
from matplotlib import cm
from torchvision.transforms.functional import to_tensor
from natsort import natsorted
FLAGS = flags.FLAGS

cprint1 = lambda s: cprint(s, 'cyan', attrs=['bold'])
cprintc = lambda s: cprint(s, 'cyan')
cprintm = lambda s: cprint(s, 'magenta')

def get_sensible_duration(duration_sec):
    """
    Format the given duration as a time string with sec/min/hrs as needed.
    E.g.: get_sensible_duration(500) -> 8.3 min
    """
    duration_min = duration_sec / 60
    duration_hrs = duration_sec / 3600
    if duration_sec <= 60:
        return '%2.1f sec' % duration_sec
    elif duration_min <= 60:
        return '%2.1f min' % duration_min
    else:
        return '%2.1f hrs' % duration_hrs

def timeit(func):
    def timed(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        cprint('%s | %s' % (func.__name__, get_sensible_duration(time.time() - t0)), 'cyan')
        return res
    return timed

def save_checkpoint(state, is_best=False, checkpoint='checkpoint', filename='checkpoint.pth.tar', filepath=None, savebestonly=False):
    if filepath is None:
        filepath = os.path.join(checkpoint, filename)
    else:
        filename = os.path.basename(filepath).split('.')[0]
        checkpoint = os.path.dirname(filepath)
    if is_best or not savebestonly:
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, '%s_best.pth.tar' % filename))

class SummaryWriter(SumWr):
    def __init__(self, comment):
        current_time = datetime.now().strftime('%b%-d_%-H-%M')
        return super(SummaryWriter, self).__init__(log_dir=os.path.join(FLAGS.tensorboard_log_dir, '_'.join([current_time, FLAGS.exp_prefix, comment])))

def beta_fig(optimizer):
    data = {i: list(p.cpu().detach().numpy()) for i, p in enumerate(optimizer.param_groups[0]['params'])}
    df = pd.DataFrame.from_dict(data, orient='index').transpose()
    sns.barplot(data=df, palette='Blues_d')
    data = [nanmedian(x) for x in data.values()]
    plt.ylim(min(0, min(data)), max(max(data) * 1.1, 2))
    ax = plt.gca()
    ax.yaxis.grid(True)
    return plt.gcf()

def fc_heatmap(fc_params, n_out_planes):
    scale = fc_params[-2].cpu().detach().numpy()
    scale = np.max(np.max(scale, axis=0).reshape(n_out_planes, -1), axis=0)
    spatial_out_dim = int(np.sqrt(len(scale)))
    sns.heatmap(scale.reshape(spatial_out_dim, spatial_out_dim), xticklabels=False, yticklabels=False)
    return plt.gcf()

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[0], columns=args[1], values=args[2])
    try:
        sns.heatmap(d, **kwargs)
    except ValueError:
        pass

def vox_rf(fc_params, n_out_planes, voxel_indices_nc_sort, dec_fc=False, mask=False):
    voxel_indices_selected = sorted(voxel_indices_nc_sort[:5])

    if mask:
        if dec_fc:
            scale_selected = np.abs(fc_params[voxel_indices_selected].cpu().detach().numpy())
        else:
            scale_selected = [np.abs(fc_params[i].cpu().detach().numpy()) for i in voxel_indices_selected]
    else:
        if dec_fc:
            W = fc_params[-2].T
        else:
            W = fc_params[-2]
        voxel_indices_selected = [i % len(W) for i in voxel_indices_selected]
        scale_selected = np.abs(W[voxel_indices_selected].cpu().detach().numpy()).reshape(len(voxel_indices_selected), n_out_planes, -1).max(axis=1)
    scale_selected = np.stack([reshape_sq(x) for x in scale_selected])

    indices = pd.MultiIndex.from_product([voxel_indices_selected] + [range(s) for s in scale_selected.shape[1:]],
                                         names=('Vox index', 'y', 'x'))
    df = pd.DataFrame(scale_selected.flatten(), index=indices, columns=('value',)).reset_index()
    fg = sns.FacetGrid(df, col='Vox index')
    fg.map_dataframe(draw_heatmap, 'x', 'y', 'value', square=True)
    for ax in fg.axes.flat:
        ax.set_aspect('equal', 'box')

    return plt.gcf()

def fc_heatmap_mask(optimizer):
    scale = optimizer.param_groups[0]['params'][0].cpu().detach().numpy()
    scale = np.max(scale, axis=0)
    sns.heatmap(scale, xticklabels=False, yticklabels=False)
    return plt.gcf()

def my_hist_comparison_fig(data, nbins):
    return hist_comparison_fig(data, np.linspace(-1.2, 1.2, nbins))

def stack2numpy(data_dict):
    return {k: torch.stack(v).numpy() for k, v in data_dict.items()}

def set_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(FLAGS.gpu)
    cprint1('(*) CUDA_VISIBLE_DEVICES: {} ({} workers per GPU)'.format(os.environ['CUDA_VISIBLE_DEVICES'], FLAGS.num_workers_gpu))

def corr_vs_corr_plot(corr_x, corr_y, ax_labels=None):
    if ax_labels is None:
        ax_labels = ['Noise ceiling', 'Pearson R (val avg)']
    df = pd.DataFrame({ax_labels[0]: corr_x, ax_labels[1]: corr_y})
    g = sns.jointplot(x=ax_labels[0], y=ax_labels[1], data=df, xlim=(0,1), ylim=(0,1), alpha=0.25)
    sns.lineplot(x=np.linspace(0,1,3), y=np.linspace(0,1,3), ax=g.ax_joint)
    return plt.gcf()

def norm_depth_01(depth):
    # Normalize to 0-1.
    # depth is NxHxW
    depth_min = depth.view(*depth.shape[:-2], -1).min(-1).values[:, None, None]
    depth_max = depth.view(*depth.shape[:-2], -1).max(-1).values[:, None, None]
    depth = (depth - depth_min) / (depth_max - depth_min)
    return depth

def tensor_transform(tensor, xfm):
    return torch.stack([xfm(x) for x in tensor])

def reconst_sbs(dec, data_loader, img_xfm_norm_inv, depth_extractor=None):
    dec.eval()
    with torch.no_grad():
        images_gt, images_D = \
            [tensor_transform(torch.cat(chunk_list, dim=0), img_xfm_norm_inv)
             for chunk_list in zip(*[(images_gt, dec(fmri_gt.cuda()).cpu().detach())
                                     for (images_gt, fmri_gt) in data_loader])]

        assert not torch.isnan(images_D).any(), 'nans in images_D.'
        # Bring GT images to same scales as the reconstructed
        images_gt_interp = interpolate(images_gt, size=images_D.size(-1), mode='bilinear')

        if depth_extractor:
            def get_depth(images):
                depth = depth_extractor(images.cuda()).cpu().detach()
                depth = norm_depth_01(depth)
                return depth.unsqueeze(1)
            images_gt_interp = torch.cat([images_gt_interp, get_depth(images_gt_interp)], 1)
            images_D = torch.cat([images_D, get_depth(images_D)], 1)

        if images_D.shape[1] != images_gt_interp.shape[1]:
            if images_D.shape[1] == 1 and images_gt_interp.shape[1] == 4:  # Depth only dec
                images_gt_interp = images_gt_interp[:, 3:]
            else:
                raise NotImplementedError

        image_D_sbs_gt = torch.cat([images_gt_interp, images_D.clamp(0, 1),
                                    ], dim=-1)
        return image_D_sbs_gt

def interpolate(x, **kwargs):
    """ Protect from Bicubic cases where pytorch gives a bug for similar in/out sizes. """
    if x is None:
        return None
    x_interp = F.interpolate(x, **kwargs)
    if x_interp.shape[-2:] == x.shape[-2:]:
        return x
    else:
        return x_interp

def gray2color(gray_tensor_HW, map_name='inferno'):
    map_name = 'jet'
    return to_tensor(cm.get_cmap(map_name)(gray_tensor_HW.numpy())[...,:3])

def create_reconst_summary(enc, dec, data_loaders_labeled, train_labeled, epoch, img_xfm_norm_inv, sum_writer, depth_extractor=None):
    enc.eval(); dec.eval()
    image_D_sbs_gt_train = reconst_sbs(
        dec, [(img_tensor.unsqueeze(0), torch.from_numpy(fmri)) for i, (img_tensor, fmri) in enumerate(train_labeled) if i in FLAGS.train_montage_indices], img_xfm_norm_inv,
        depth_extractor)
    # NxCxHxW
    if sum_writer:
        grid_img = make_grid(image_D_sbs_gt_train, nrow=5, padding=3)
        if len(image_D_sbs_gt_train[0]) >= 3:
            sum_writer.add_image('TestDec/TrainReconst', grid_img[:3], epoch)
        if len(grid_img) == 4 or len(image_D_sbs_gt_train[0]) == 1:
            if len(grid_img) == 4: # RGBD
                grid_img_depth = grid_img[3]
            else:  # Depth only
                grid_img_depth = grid_img[0]
            sum_writer.add_image('TestDec/TrainReconstDepth', gray2color(grid_img_depth), epoch)

    image_D_sbs_gt_test = reconst_sbs(dec, data_loaders_labeled['test'], img_xfm_norm_inv, depth_extractor)
    if sum_writer:
        grid_img = make_grid(image_D_sbs_gt_test, nrow=5, padding=3)
        if len(image_D_sbs_gt_test[0]) >= 3:
            sum_writer.add_image('TestDec/Reconst', grid_img[:3], epoch)
        if len(grid_img) == 4 or len(image_D_sbs_gt_test[0]) == 1:
            if len(grid_img) == 4: # RGBD
                grid_img_depth = grid_img[3]
            else:  # Depth only
                grid_img_depth = grid_img[0]
            sum_writer.add_image('TestDec/ReconstDepth', gray2color(grid_img_depth), epoch)

    return image_D_sbs_gt_train, image_D_sbs_gt_test

def extract_patches(x, patch_size):
    patches = x.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    # Permute so that channels are next to patch dimension
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [128, 32, 32, 16, 3, 3]
    # View as [batch_size, height, width, channels*kh*kw]
    patches = patches.view(*patches.size()[:3], -1)
    # View as BxCxHxW
    patches = patches.permute(0, 3, 1, 2).contiguous()
    return patches

def tup2list(tuple_list, tuple_idx):
    return list(zip(*tuple_list))[tuple_idx]

def chained(l):
    return list(itertools.chain(*l))

def param_count_str(parameters):
    N = sum(p.numel() for p in parameters)
    if N > 1e6:
        return '%.2fM' % (N / 1000000.0)
    elif N > 1e3:
        return '%.1fK' % (N / 1000.0)
    else:
        return '%d' % N

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, relative=False):
        self.relative = relative
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.relative:
            if not self.count:
                self.scale = 100 / abs(val)
            val *= self.scale
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def hw_flatten(tensor):
    return tensor.view(*tensor.shape[:-2], -1)

def batch_flat(tensor):
    return tensor.view(len(tensor), -1)

def sample_portion(x, q):
    return x[np.random.permutation(len(x))[:int(len(x) * q)]]

def hist_comparison_fig(dist_dict, bins, **flags):
    for k, v in dist_dict.items():
        plt.hist(v, bins, alpha=0.5, label=k, **flags)
    plt.legend()
    return plt.gcf()

def reshape_sq(a):
    d = np.sqrt(np.prod(a.shape))
    assert int(d) - d == 0, 'Cannot make array to a square'
    d = int(d)
    return a.reshape(d, d)

def silentremove(file_or_folder_name):
    if os.path.exists(file_or_folder_name):
        if os.path.isfile(file_or_folder_name):
            os.remove(file_or_folder_name)
        else:  # Folder
            shutil.rmtree(file_or_folder_name)

def overridefolder(folder_path):
    silentremove(folder_path)
    os.makedirs(folder_path)
    return folder_path


if __name__ == '__main__':
    pass
