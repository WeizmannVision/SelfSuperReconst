import os, time

from .misc import *
from .datasets import *
from .criterion import *
from .models import *

sns.set(color_codes=True)

def starred(s, n_stars=10):
    return '*' * n_stars + '\n' + s + '\n' + '*' * n_stars

class NormalizeBatch(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if tensor.ndim == 4:
            tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        elif tensor.ndim == 3:
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        else:
            raise TypeError('tensor is not a torch image nor an image batch.')
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class NormalizeImageNet(transforms.Normalize):
    def __init__(self):
        super(NormalizeImageNet, self).__init__(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

normalize_imagenet = NormalizeImageNet()

norm_depth_img = lambda img_depth: (img_depth - img_depth.mean()) / max(img_depth.std(), .1)

norm_batch_imagenet = NormalizeBatch(NormalizeImageNet().mean, NormalizeImageNet().std)

norm_imagenet_norm_depth_img = lambda img: torch.cat([normalize_imagenet(img[:3]), norm_depth_img(img[3:])])

def norm_batch_within_img(tensor_NCHW):
    tensor_NCHW_mean = tensor_NCHW.view(*tensor_NCHW.shape[:-2], -1).mean(-1)[:, :, None, None]
    tensor_NCHW_std = tensor_NCHW.view(*tensor_NCHW.shape[:-2], -1).std(-1)[:, :, None, None]
    return (tensor_NCHW - tensor_NCHW_mean) / tensor_NCHW_std.clamp(min=.1)

def norm_imagenet_norm_depth_img_batch(tensor): 
    if tensor.shape[1] in [3, 4]:
        tensor_rgb_norm = norm_batch_imagenet(tensor[:, :3])
        if tensor.shape[1] == 4:
            tensor_depth_norm = norm_batch_within_img(tensor[:, 3:])
            return torch.cat([tensor_rgb_norm, tensor_depth_norm] , 1)
        else:
            return tensor_rgb_norm
    elif tensor.shape[1] == 1:
        return norm_batch_within_img(tensor[:, :1])
    else:
        raise NotImplementedError
