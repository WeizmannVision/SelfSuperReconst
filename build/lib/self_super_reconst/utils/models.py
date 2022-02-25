"""
    Custom models.
    Date created: 8/25/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

import self_super_reconst
from self_super_reconst.utils.datasets import RGBD_Dataset
from self_super_reconst.utils.misc import (cprint1, cprintc, cprintm, np,
                                           interpolate, extract_patches,
                                           norm_depth_01, tup2list, hw_flatten)
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import pretrainedmodels as pm
import torchvision.models as torchvis_models
from pretrainedmodels import utils as pmutils
from self_super_reconst.config import *
from self_super_reconst.midas.midas_net import MidasNet
from self_super_reconst.midas.midas_net_custom import MidasNet_small
from absl import flags
identity = lambda x: x

FLAGS = flags.FLAGS

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class MultiBranch(nn.Module):
    def __init__(self, model, branch_dict, main_branch, spatial_out_dims=20, replace_maxpool=False):
        super(MultiBranch, self).__init__()
        name_to_module = dict(model.named_modules())
        self.branch_dict = branch_dict
        self.target_modules = list(branch_dict.values())
        self.main_branch = main_branch
        self.adapt_avg_pool_suffix = '_adapt_avg_pool'
        if spatial_out_dims is not None and isinstance(spatial_out_dims, int):
            spatial_out_dims = dict(zip(self.target_modules, [spatial_out_dims] * len(self.target_modules)))

        for module_name in main_branch:
            module = name_to_module[module_name]
            if replace_maxpool and isinstance(module, nn.MaxPool2d):
                module = nn.Upsample(scale_factor=.5)
            self.add_module(module_name.replace('.', '_'), module)
        for module_name in self.target_modules:
            if spatial_out_dims is not None:
                module = nn.AdaptiveAvgPool2d(spatial_out_dims[module_name])
                self.add_module(module_name + self.adapt_avg_pool_suffix, module)

    def __getitem__(self, module_name):
        return getattr(self, module_name.replace('.', '_'), None)

    def num_output_planes(self):
        n_planes = []
        for target_module in self.target_modules:
            for module_name in self.main_branch[:self.main_branch.index(target_module)+1][::-1]:
                try:
                    n_planes.append(list(self[module_name].parameters())[0].shape[0])
                    break
                except:
                    pass
        return n_planes

    def forward(self, x):
        X = {}
        for module_name in self.main_branch:
            if isinstance(self[module_name], nn.Linear) and x.ndim > 2:
                x = x.view(len(x), -1)
            x = self[module_name](x)
            if module_name in self.target_modules: # Collect
                X[module_name] = x.clone()
                avg_pool = self[module_name + self.adapt_avg_pool_suffix]
                if avg_pool:
                    X[module_name] = avg_pool(X[module_name])
        return list(X.values())

class BaseEncoderVGG19ml(nn.Module):
    def __init__(self, out_dim, random_crop_pad_percent, spatial_out_dim=None, drop_rate=0.25):
        super(BaseEncoderVGG19ml, self).__init__()
        self.drop_rate = drop_rate
        cprintm('(*) Backbone: {}'.format('vgg19'))
        bbn = pm.__dict__['vgg19'](num_classes=1000, pretrained='imagenet')
        self.img_xfm_basic = pmutils.TransformImage(bbn, scale=1)
        self.img_xfm_train = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BILINEAR),
            transforms.RandomCrop(size=224, padding=int(random_crop_pad_percent / 100 * 224), padding_mode='edge'),
            *self.img_xfm_basic.tf.transforms[-4:],
        ])

        branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
            'conv1': ['_features.{}'.format(i) for i in range(4)],
            'conv2': ['_features.{}'.format(i) for i in range(9)],
            'conv3': ['_features.{}'.format(i) for i in range(18)],
            'conv4': ['_features.{}'.format(i) for i in range(27)],
        }

        spatial_out_dims = None

        main_branch = list(branch_dict.values())[-1]
        branch_dict = {layer: branch_module_list[-1] for layer, branch_module_list in branch_dict.items()}
        self.multi_branch_bbn = MultiBranch(bbn, branch_dict, main_branch, spatial_out_dims=spatial_out_dims)

        self.bbn_n_out_planes = self.multi_branch_bbn.num_output_planes()
        self.out_shapes = [(48, 14, 14)]
        self.n_out_planes = self.out_shapes[0][0]
        in_dim = np.prod(self.out_shapes[0])

        kernel_size = 3
        pad_size = int(kernel_size // 2)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[0]),
            nn.MaxPool2d(2),

            nn.Conv2d(self.bbn_n_out_planes[0], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),

            nn.Conv2d(self.n_out_planes, self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[1]),

            nn.Conv2d(self.bbn_n_out_planes[1], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),

            nn.Conv2d(self.n_out_planes, self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[2]),

            nn.Conv2d(self.bbn_n_out_planes[2], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[3]),

            nn.Conv2d(self.bbn_n_out_planes[3], self.n_out_planes, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )
        self.bn_sum = nn.BatchNorm2d(self.n_out_planes)
        self.dropout = nn.Dropout(drop_rate)
        self.fc_head = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.trainable = self.convs + [self.bn_sum, self.fc_head]


    def forward_bbn(self, x, detach_bbn=False):
        x = interpolate(x, size=im_res(), mode=FLAGS.interp_mode)
        X = self.multi_branch_bbn(x)
        if detach_bbn:
            X = [xx.detach() for xx in X]
        feats_dict = dict(zip(self.multi_branch_bbn.branch_dict.keys(), X))
        return feats_dict

    def forward_convs(self, feats_dict):
        X = [conv(xx) for xx, conv in zip(feats_dict.values(), self.convs)]
        return X

    def forward(self, x, feats=False, detach_bbn=False):
        feats_dict = self.forward_bbn(x, detach_bbn=detach_bbn)
        X = self.forward_convs(feats_dict)
        x = torch.stack(X).sum(0)
        x = self.bn_sum(x)
        x = self.dropout(x)
        x = self.fc_head(x.view(x.size(0), -1))
        if feats:
            return x, feats_dict
        else:
            return x

class SeparableEncoderVGG19ml(nn.Module):
    def __init__(self, out_dim, random_crop_pad_percent, spatial_out_dim=None, drop_rate=0.25):
        super(SeparableEncoderVGG19ml, self).__init__()
        self.drop_rate = drop_rate
        cprintm('(*) Backbone: {}'.format('vgg19'))

        if FLAGS.is_rgbd:
            bbn = torchvis_models.__dict__['vgg19'](pretrained=True)
        else:
            bbn = pm.__dict__['vgg19'](num_classes=1000, pretrained='imagenet')

        if FLAGS.is_rgbd:
            if FLAGS.is_rgbd == 1:  # RGBD
                bbn.features = nn.Sequential(
                    nn.Conv2d(4, 64, 3, padding=1),
                    *bbn.features[1:])
                ckpt_name = 'vgg19_rgbd_large_norm_within_img'
            else:  # Depth only
                bbn.features = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    *bbn.features[1:])
                ckpt_name = 'vgg19_depth_only_large_norm_within_img'

            cprint1('   >> Loading Encoder bbn checkpoint: {}'.format(ckpt_name))
            state_dict_loaded = torch.load(f'{PROJECT_ROOT}/data/imagenet_rgbd/{ckpt_name}_best.pth.tar')['state_dict']
            state_dict_loaded = { k.replace('module.', ''): v for k, v in state_dict_loaded.items() }
            bbn.load_state_dict(state_dict_loaded)

            branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
                'conv1': ['features.{}'.format(i) for i in range(4)],
                'conv2': ['features.{}'.format(i) for i in range(9)],
                'conv3': ['features.{}'.format(i) for i in range(18)],
                'conv4': ['features.{}'.format(i) for i in range(27)],
            }
        else:
            branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
                'conv1': ['_features.{}'.format(i) for i in range(4)],
                'conv2': ['_features.{}'.format(i) for i in range(9)],
                'conv3': ['_features.{}'.format(i) for i in range(18)],
                'conv4': ['_features.{}'.format(i) for i in range(27)],
            }

        spatial_out_dims = None
        main_branch = list(branch_dict.values())[-1]
        branch_dict = {layer: branch_module_list[-1] for layer, branch_module_list in branch_dict.items()}
        self.multi_branch_bbn = MultiBranch(bbn, branch_dict, main_branch, spatial_out_dims=spatial_out_dims)

        self.bbn_n_out_planes = self.multi_branch_bbn.num_output_planes()
        self.patch_size = 3
        self.out_shapes = [(32, 28 - self.patch_size + 1, 28 - self.patch_size + 1)] * 3 + [(32, 14 - self.patch_size + 1, 14 - self.patch_size + 1)]
        self.n_out_planes = self.out_shapes[0][0]
        kernel_size = 3
        pad_size = int(kernel_size // 2)

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[0]),

            nn.MaxPool2d(2),

            nn.Conv2d(self.bbn_n_out_planes[0], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[1]),

            nn.Conv2d(self.bbn_n_out_planes[1], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[2]),

            nn.Conv2d(self.bbn_n_out_planes[2], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[3]),

            nn.Conv2d(self.bbn_n_out_planes[3], self.n_out_planes, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

        # Separable part
        self.space_maps = nn.ModuleDict({
            str(in_space_dim): nn.Linear(in_space_dim**2, out_dim, bias=False) for in_space_dim in np.unique(tup2list(self.out_shapes, 1))
        })

        self.chan_mixes = nn.ModuleList([ChannelMix(self.n_out_planes*self.patch_size**2, out_dim) for _ in range(len(self.convs))])
        self.branch_mix = nn.Parameter(torch.Tensor(out_dim, len(self.chan_mixes)))
        self.branch_mix.data.fill_(1.)

        self.dropout = nn.Dropout(drop_rate)

        self.trainable = self.convs + list(self.space_maps.values()) + list(self.chan_mixes) + [self.branch_mix]

    def forward_bbn(self, x, detach_bbn=False):
        x = interpolate(x, size=im_res(), mode=FLAGS.interp_mode)
        X = self.multi_branch_bbn(x)
        if detach_bbn:
            X = [xx.detach() for xx in X]
        feats_dict = dict(zip(self.multi_branch_bbn.branch_dict.keys(), X))
        return feats_dict

    def forward_convs(self, feats_dict):
        X = [conv(xx) for xx, conv in zip(feats_dict.values(), self.convs)]
        return X

    def forward(self, x, feats=False, detach_bbn=False):
        feats_dict = self.forward_bbn(x, detach_bbn=detach_bbn)
        X = self.forward_convs(feats_dict)
        X = [extract_patches(x, self.patch_size) for x in X]
        X = [self.space_maps[str(x.shape[-1])](hw_flatten(x)) for x in X]  # => BxCxV
        X = [self.dropout(x) for x in X]
        X = [f(x) for f, x in zip(self.chan_mixes, X)]

        x = torch.stack(X, dim=-1)
        x = (x * self.branch_mix.abs()).sum(-1)
        if feats:
            return x, feats_dict
        else:
            return x

class ChannelMix(nn.Module):
    def __init__(self, n_chan, out_dim):
        super(ChannelMix, self).__init__()
        self.chan_mix = nn.Parameter(torch.Tensor(out_dim, n_chan))
        nn.init.xavier_normal(self.chan_mix)

        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.bias.data.fill_(0.01)

    def forward(self, x):
        # BxCxN
        x = (x * self.chan_mix.T).sum(-2)
        x += self.bias
        # BxN
        return x

class BaseDecoder(nn.Module):
    def __init__(self, in_dim, out_img_res, start_CHW=(64, 14, 14), n_conv_layers_ramp=3, n_chan=64, n_chan_output=3, depth_extractor=None):
        super(BaseDecoder, self).__init__()

        self.start_CHW = start_CHW
        upsample_scale_factor = (out_img_res / start_CHW[-1]) ** (1/n_conv_layers_ramp)
        self.input_fc = nn.Linear(in_dim, np.prod(self.start_CHW))

        kernel_size = 5

        pad_size = int(kernel_size // 2)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=upsample_scale_factor, mode=FLAGS.interp_mode),
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(start_CHW[0], n_chan, kernel_size),
            nn.GroupNorm(32, n_chan),
            MemoryEfficientSwish(),
        ) for block_index in range(n_conv_layers_ramp)] + \
        [nn.Sequential(
            nn.Conv2d(start_CHW[0], n_chan, kernel_size, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_chan)
        ) for _ in range(0)])

        self.top = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(n_chan, n_chan_output, kernel_size),
            nn.Sigmoid()
        )

        self.depth_extractor = depth_extractor

        self.trainable = [self.input_fc, self.blocks, self.top]

    def forward(self, x):
        x = self.input_fc(x)
        x = x.view(-1, *self.start_CHW)

        for block_index, block in enumerate(self.blocks):
            x = block(x)

        x = self.top(x)

        if self.depth_extractor:
            x_depth = self.depth_extractor(x)
            x_depth = norm_depth_01(x_depth).unsqueeze(1)
            x = torch.cat([x, x_depth], 1)

        return x

class DepthExtractor(nn.Module):
    def __init__(self, img_xfm_norm=identity, model_type=None):
        super(DepthExtractor, self).__init__()
        if not model_type:
            model_type = FLAGS.midas_type
        if model_type == "large":
            model_path = f'{PROJECT_ROOT}/data/model-f6b98070.pt'
            self.model = self_super_reconst.midas.midas_net.MidasNet(model_path, non_negative=True)
            self.net_input_size = 384
        elif model_type == "small":
            model_path = f'{PROJECT_ROOT}/data/model-small-70d6b9c8.pt'
            self.model = self_super_reconst.midas.midas_net_custom.MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            self.net_input_size = 256
        self.model.eval()
        self.img_xfm_norm = img_xfm_norm

    def forward(self, x):
        orig_size = x.shape[-1]
        x = interpolate(x, size=self.net_input_size, mode=FLAGS.interp_mode)
        x = self.img_xfm_norm(x)
        pred = self.model.forward(x)
        pred = interpolate(pred.unsqueeze(1), size=orig_size, mode=FLAGS.interp_mode).squeeze(1)
        # Normalize
        pred = (pred - pred.view(len(pred), -1).mean(1)[:, None, None]) / (pred.view(len(pred), -1).std(1)[:, None, None] + 1e-4)
        return pred  # NxHxW

def make_model(model_type, *args, **kwargs):
    return globals()[model_type](*args, **kwargs)
