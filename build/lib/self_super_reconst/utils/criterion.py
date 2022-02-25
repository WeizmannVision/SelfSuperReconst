"""
    Evaluation metrics between predicted and ground-truth data.
    Date created: 8/27/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

from self_super_reconst.utils.misc import batch_flat, np
import torch
from torch import nn
from self_super_reconst.utils.models import *
import torchvision.models as torchvis_models
from self_super_reconst.config import PROJECT_ROOT


def pearson_corr(feats_A, feats_B):
    '''
    Assume structure NxK and returns K correlations for each of the features.
    '''
    feats_A = feats_A - torch.mean(feats_A, 0, keepdim=True)
    feats_B = feats_B - torch.mean(feats_B, 0, keepdim=True)
    overflow_factor = (batch_flat(feats_A**2).sum(1) + batch_flat(feats_B**2).sum(1)).mean()/2
    r = torch.diag((feats_A.transpose(1, 0) / overflow_factor) @ (feats_B / overflow_factor)) * overflow_factor**2 / \
        (torch.sqrt(torch.sum(feats_A ** 2, 0)) * torch.sqrt(torch.sum(feats_B ** 2, 0)))
    return r

def pearson_corr_piecewise(feats_A, feats_B, win_size=None):
    if win_size is None:
        win_size = len(feats_A)
    def corr(a, b):
        a = a - torch.mean(a, 0, keepdim=True)
        b = b - torch.mean(b, 0, keepdim=True)
        r = torch.diag(a.transpose(1, 0) @ b)
        return r
    def std(x):
        return ((x - torch.mean(x, 0, keepdim=True)) ** 2).sum(0).sqrt()
    numer = sum([corr(feats_A[i: i+win_size], feats_B[i: i+win_size]) for i in range(len(feats_A)-win_size)])
    denom = sum([std(feats_A[i: i+win_size]) * std(feats_B[i: i+win_size]) for i in range(len(feats_A)-win_size)])
    return numer / denom

def total_variation(x, p=2):
    return ((x[..., 1:, :-1] - x[...,1:, 1:])**2 +
           (x[..., :-1, 1:] - x[..., 1:, 1:])**2).pow(p/2).mean()

def cosine_loss(pred, actual, return_mean=True):
    cos_sim = F.cosine_similarity(pred.view(len(pred), -1), actual.view(len(pred), -1), dim=1)
    cos_loss = (1-cos_sim) / 2
    if return_mean:
        return cos_loss.mean()
    else:
        return cos_loss

def normalize_channel(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True) + eps)
    return in_feat/(norm_factor + eps)

def perceptual_loss_layer(pred, actual, mask=1):
    pred, actual = map(normalize_channel, [pred, actual])
    return ((1 - (pred * actual).sum(1)) * mask).mean()

def gradient(tensor_NCHW):
        return tensor_NCHW[:, :, :, 1:] - tensor_NCHW[:, :, :, :-1], tensor_NCHW[:, :, 1:, :] - tensor_NCHW[:, :, :-1, :]

def euclidian_loss(org_matrix, target_matrix):
    """
        Euclidian loss is the main loss function in the paper
        ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2
    """
    distance_matrix = target_matrix - org_matrix
    euclidian_distance = alpha_norm(distance_matrix, 2)
    normalized_euclidian_distance = euclidian_distance / alpha_norm(org_matrix, 2)
    return normalized_euclidian_distance

def norm_tensor_batchwise(tensor):
    return tensor / tensor.view(len(tensor), -1).abs().sum(1).mean()

class ImageLoss(nn.Module):
    def __init__(self, feats_extractor=None, img_xfm_norm=identity, data_norm_factors_images=None):
        super(ImageLoss, self).__init__()
        if feats_extractor is None:
            bbn = torchvis_models.__dict__['vgg16'](pretrained=True)

            if FLAGS.is_rgbd or FLAGS.rgbd_loss:
                if (FLAGS.is_rgbd == 1 or FLAGS.rgbd_loss) and not FLAGS.depth_dec:  # RGBD
                    bbn.features = nn.Sequential(
                        nn.Conv2d(4, 64, 3, padding=1),
                        *bbn.features[1:])
                    ckpt_name = 'vgg16_rgbd_large_norm_within_img'
                else:  # Depth only
                    bbn.features = nn.Sequential(
                        nn.Conv2d(1, 64, 3, padding=1),
                        *bbn.features[1:])
                    ckpt_name = 'vgg16_depth_only_large_norm_within_img'

                cprint1('   >> Loading ImageLoss bbn checkpoint: {}'.format(ckpt_name))
                state_dict_loaded = torch.load(f'{PROJECT_ROOT}/data/imagenet_rgbd/{ckpt_name}_best.pth.tar')['state_dict']
                state_dict_loaded = { k.replace('module.', ''): v for k, v in state_dict_loaded.items() }
                bbn.load_state_dict(state_dict_loaded)

            branch_dict = {  # VGG16 Blocks  # selectedLayers = [3, 6, 10, 14, 18]
                # After maxpools
                'conv1': ['features.{}'.format(i) for i in range(5)],
                'conv2': ['features.{}'.format(i) for i in range(10)],
                'conv3': ['features.{}'.format(i) for i in range(17)],
                'conv4': ['features.{}'.format(i) for i in range(24)],
                'conv5': ['features.{}'.format(i) for i in range(31)],
            }

            spatial_out_dims = None
            main_branch = branch_dict['conv5']
            branch_dict = {layer: branch_module_list[-1] for layer, branch_module_list in branch_dict.items()}
            self.feats_extractor = MultiBranch(bbn, branch_dict, main_branch, spatial_out_dims=spatial_out_dims)
        else:
            self.feats_extractor = feats_extractor
        self.feats_extractor.eval()
        self.img_xfm_norm = img_xfm_norm

        self.feats_extractor_wrap = lambda x: self.feats_extractor(img_xfm_norm(x))

        self.branch_weights_cos = []

        if data_norm_factors_images is not None:
            cprint1('   >> Computing normalization factors.')
            self.feats_extractor.eval()
            with torch.no_grad():
                images_tensor = torch.stack([x for x in data_norm_factors_images])
                norm_factors_channel = [nn.Parameter(images_tensor.abs().mean(-1).mean(-1).mean(0)[None, :, None, None].clamp_(min=1e-6), requires_grad=False)]
                norm_factors_channel += [nn.Parameter(x.detach().abs().mean(-1).mean(-1).mean(0)[None, :, None, None].clamp_(min=1e-6), requires_grad=False) \
                    for x in self.feats_extractor_wrap(images_tensor)[:-1]]  # The [:-1] is to ensure that class layer is not included
        else:
            norm_factors_channel = [nn.Parameter(torch.ones(1, 3, 1, 1,dtype=torch.float), requires_grad=False) for _ in range(len(branch_dict))]

        self.norm_factors_channel = nn.ParameterDict(dict(zip(['image'] + list(branch_dict.keys())[:-1], norm_factors_channel)))

        if FLAGS.midas_loss > 0 or FLAGS.rgbd_loss:
            self.depth_extractor = DepthExtractor(img_xfm_norm=NormalizeBatch(NormalizeImageNet().mean, NormalizeImageNet().std) if FLAGS.is_dec01 else identity)
            self.depth_extractor_wrap = lambda x: norm_depth_01(self.depth_extractor(x)).unsqueeze(1)

    def forward(self, pred, actual, sum_writer=None):
        if FLAGS.pred_interp > 0:
            pred = interpolate(pred, size=FLAGS.pred_interp, mode=FLAGS.interp_mode)
        actual = interpolate(actual, size=pred.size(-1), mode=FLAGS.interp_mode)

        if FLAGS.depth_dec and actual.shape[1] > 1:  # discard all channels other than depth
            actual = actual[:, 3:]

        if FLAGS.rgbd_loss:
            with torch.no_grad():
                actual_depth = self.depth_extractor_wrap(actual).detach()
            pred_depth = self.depth_extractor_wrap(pred)
            actual = torch.cat([actual, actual_depth], 1)
            pred = torch.cat([pred, pred_depth], 1)

        if FLAGS.midas_loss > 0:  # midas loss
            with torch.no_grad():
                actual_depth = self.depth_extractor(actual).detach()
            midas_loss = F.l1_loss(self.depth_extractor(pred), actual_depth)
        else:
            midas_loss = 0

        if FLAGS.is_rgbd in [0, 1]:
            loss_rgb_mae = F.l1_loss(*[(x / self.norm_factors_channel['image'])[:, :3] for x in [pred, actual]])
        else:
            loss_rgb_mae = 0

        # Depth
        if FLAGS.is_rgbd or FLAGS.rgbd_loss:
            if (FLAGS.is_rgbd == 1 or FLAGS.rgbd_loss) and not FLAGS.depth_dec:  # RGBD
                loss_depth_mae = F.l1_loss(*[(x / self.norm_factors_channel['image'])[:, 3:] for x in [pred, actual]])
            else:  # Depth only
                loss_depth_mae = F.l1_loss(*[(x / self.norm_factors_channel['image'])[:, :1] for x in [pred, actual]])
        else:
            loss_depth_mae = 0

        loss_feats_dict = {}
        with torch.no_grad():
            actual_feats_list = [x.detach() for x in self.feats_extractor_wrap(actual)]

        pred_feats_list = [x for x in self.feats_extractor_wrap(pred)]
        for layer, (pred_feats, actual_feats) in zip(self.feats_extractor.branch_dict.keys(), zip(pred_feats_list, actual_feats_list)):
            loss_feats_dict[layer] = {'perceptual': perceptual_loss_layer(pred_feats, actual_feats)}

        for w_cos, branch_name in zip(self.branch_weights_cos, loss_feats_dict.keys()):
            if 'cosine' in loss_feats_dict[branch_name]:
                loss_feats_dict[branch_name]['cosine'] *= w_cos

        tv_loss = total_variation(pred, p=2*1.25)

        loss_list = [('rgb_mae', FLAGS.rgb_mae * loss_rgb_mae), \
            ('midas', FLAGS.midas_loss * midas_loss)] + \
            [ \
                (f'mlpercep_layerconv{conv_i}', w * loss_feats_dict[f'conv{conv_i}']['perceptual']) for conv_i, w in zip([1,2,3,4,5], [int(x) for x in FLAGS.percept_w]) \
            ] + \
            [('tv', FLAGS.tv_reg * tv_loss)] + \
            [('depth_mae', FLAGS.depth_mae * loss_depth_mae)]

        return loss_list

class FmriLoss(nn.Module):
    def __init__(self):
        super(FmriLoss, self).__init__()
        self.losses_dict = {
            1.: nn.L1Loss(),
            0.1: cosine_loss
        }

    def forward(self, pred, actual):
        return sum([w * loss_func(pred, actual) for w, loss_func in self.losses_dict.items()])

if __name__ == '__main__':
    pass
