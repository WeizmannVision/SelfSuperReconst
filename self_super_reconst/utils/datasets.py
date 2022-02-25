"""
    Customized datasets.
    Date created: 8/25/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

from self_super_reconst.utils.misc import (cprint1, cprintc, cprintm, os, np,
                                           Image, natsorted, glob)
import itertools
import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from self_super_reconst.config import PROJECT_ROOT
identity = lambda x: x
from absl.flags import FLAGS

class KamitaniDataset(Dataset):
    TRAIN, TEST, TEST_AVG = range(3)

    def __init__(self, roi, sbj_num=3, im_res=112, img_xfm=identity, fmri_xfm=identity, subset_case=TRAIN, test_avg_qntl=.33, select_voxels='', is_rgbd=0):
        super(KamitaniDataset, self).__init__()
        if is_rgbd:
            images_data = np.load(f'{PROJECT_ROOT}/data/rgbd_{im_res}_from_224_large_png_uint8.npz')
            if is_rgbd == 1:
                cprintc('(*) RGBD')
            else:  # Depth only
                cprintc('(*) Depth only')
                images_data = {k: images[..., 3:] for k, images in images_data.items()}
        else:
            images_data = np.load(f'{PROJECT_ROOT}/data/images_{im_res}.npz')

        dict_loaded = dict(np.load(f'{PROJECT_ROOT}/data/sbj_{sbj_num}.npz'))
        Y, Y_test, Y_test_avg, labels_train, test_labels, vox_noise_ceil, vox_snr = [dict_loaded.pop(f'arr_{k}') for k in range(7)]
        roi_masks = dict_loaded

        # Y, Y_test, Y_test_avg, labels_train, test_labels, vox_noise_ceil, vox_snr, roi_masks = list(dict(np.load(f'data/sbj_{sbj_num}.npz')).values())

        cprintc(f'(*) ROI: {roi}')
        roi_mask = roi_masks[roi]
        # Screen by mask
        Y = Y[..., roi_mask]
        Y_test = Y_test[..., roi_mask]
        Y_test_avg = Y_test_avg[..., roi_mask]
        vox_noise_ceil = vox_noise_ceil[roi_mask]
        vox_snr = vox_snr[roi_mask]

        if select_voxels is not None and len(select_voxels):
            cprintc(f'(*) Selecting voxels by {select_voxels}')
            select_voxels = np.load(select_voxels)['select_voxels']
            Y, Y_test, Y_test_avg, vox_noise_ceil, vox_snr = map(lambda X: X[..., select_voxels], [Y, Y_test, Y_test_avg, vox_noise_ceil, vox_snr])

        self.voxel_score = dict(noise_ceil=vox_noise_ceil, snr=vox_snr)

        self.n_voxels = Y.shape[-1]
        if subset_case == self.TRAIN:
            self.images = images_data['train_images'][labels_train]
            self.fmri = Y
        elif subset_case == self.TEST:
            if test_avg_qntl == 0:
                self.images = images_data['test_images'][test_labels]
                self.fmri = Y_test
            else:
                self.images = images_data['test_images']
                self.fmri = separate_repeats(Y_test, test_labels)
                self.n_avg = int(self.fmri.shape[1] * test_avg_qntl)
                cprintm('(+) Averaging {} test repeats at random.'.format(self.n_avg))
                assert self.n_avg > 0
        elif subset_case == self.TEST_AVG:
            self.images = images_data['test_images']
            self.fmri = Y_test_avg
        else:
            raise NotImplementedError
        assert len(self.images) == len(self.fmri)
        self.img_xfm = img_xfm
        self.fmri_xfm = fmri_xfm
        self.mix_map = list(itertools.combinations(np.arange(len(self.images)), 1))

    def __getitem__(self, index):
        items = map(self.getitem, self.mix_map[index])
        img, fmri = map(lambda x: np.mean(np.stack(x), axis=0), zip(*items))
        img = self.img_xfm(Image.fromarray((img.squeeze() * 255.).astype('uint8')))
        fmri = self.fmri_xfm(fmri)
        return img, fmri

    def getitem(self, index):
        img = self.images[index]
        fmri = self.fmri[index]
        if hasattr(self, 'n_avg'):
            fmri = sample_array(fmri, size=self.n_avg).mean(axis=0)

        return img, fmri

    def __len__(self):
        return len(self.mix_map)

    def get_voxel_score(self, score_type='noise_ceil'):
        return self.voxel_score[score_type]

class RGBD_Dataset(Dataset):
    def __init__(self, depth_only=False):
        super(RGBD_Dataset, self).__init__()
        self.rgb_images = ImageFolder(f'{PROJECT_ROOT}/data/imagenet/val')
        self.depth_images_paths = natsorted(glob.glob(f'{PROJECT_ROOT}/data/imagenet_depth/val_depth_on_orig_small_png_uint8/*'))
        self.depth_only = depth_only
        assert len(self.rgb_images) == len(self.depth_images_paths), "Cannot resolve correspondence of rgb images and depth maps (diff length)."

    def __getitem__(self, index):
        # IMPORTANT: assume correspondence between the RGB images and their depth maps
        img_rgb, class_label = self.rgb_images[index]
        img_rgb = np.array(img_rgb)
        img_depth = np.array(Image.open(self.depth_images_paths[index]))[...,None]
        if self.depth_only:
            return Image.fromarray(img_depth.squeeze()), class_label
        else:
            return Image.fromarray(np.dstack([img_rgb, img_depth])), class_label

    def __len__(self):
        return len(self.rgb_images)

class CustomDataset(Dataset):
    def __init__(self, dataset, input_xfm=identity, output_xfm=identity):
        self.dataset = dataset
        self.input_xfm = input_xfm
        self.output_xfm = output_xfm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, output = self.dataset[index]
        return self.input_xfm(input), self.output_xfm(output)

class UnlabeledDataset(Dataset):
    def __init__(self, dataset, return_tup_index=0):
        self.dataset = dataset
        self.return_tup_index = return_tup_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][self.return_tup_index]

def separate_repeats(test_data, test_lbl):
    ''' First returned dims are NxR '''
    return np.stack([test_data[test_lbl == lbl] for lbl in np.unique(test_lbl)])

def sample_array(a, axis=0, size=1, replace=False):
    a = np.array(a)
    if replace:
        indices = np.array(random.choices(range(a.shape[axis]), k=size))
    else:
        indices = np.random.permutation(a.shape[axis])[:size]
    return np.take(a, indices, axis=axis)