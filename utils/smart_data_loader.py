import numpy as np
import torch
from torch.utils import data
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from utils.data_loader_create_tilling import generate_tiling_for_data_loader
from utils.data_aug import transformation


class Data(data.Dataset):
    def __init__(self, large_image_path, large_gt_path, large_mask_path, w_size, data_aug=None, aug_mode=None, dilation=False):
        self.image_path = large_image_path
        self.gt_path = large_gt_path
        self.mask_path = large_mask_path
        self.w_size = w_size
        self.data_aug = data_aug
        self.aug_mode = aug_mode
        self.dilation = dilation
        self.image_tiles = np.array(generate_tiling_for_data_loader(self.image_path, w_size=self.w_size))
        self.gt_tiles = np.array(generate_tiling_for_data_loader(self.gt_path, w_size=self.w_size))
        self.mask_tiles = np.array(generate_tiling_for_data_loader(self.mask_path, w_size=self.w_size))
        print('Window_size: {}, Generate {} image patches, {} gt patches and {} mask patches.'.format(
            w_size, len(self.image_tiles), len(self.gt_tiles), len(self.mask_tiles)))

    def __len__(self):
        return len(self.image_tiles)

    def __getitem__(self, index):
        img = self.image_tiles[index]
        labels = self.gt_tiles[index]
        mask= self.mask_tiles[index]

        if self.dilation:
            struct1 = ndimage.generate_binary_structure(2, 2)
            labels = binary_dilation(labels, structure=struct1).astype(np.uint8)

        if self.data_aug:
            img, labels, mask = transformation(img, labels, mask, self.aug_mode)

        img = img / 255.
        img = np.array(img, dtype=np.float32)

        labels = labels/255.
        labels = labels.astype(np.uint8)

        mask = mask / 255.
        mask = mask.astype(np.uint8)

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        labels = torch.from_numpy(np.array([labels])).float()
        mask = torch.from_numpy(mask).float()

        return img, labels, mask, index


'''
class Data2(data.Dataset):
    def __init__(self, large_image_path, large_gt_path, large_mask_path, w_size, data_aug=None, aug_mode=None, dilation=False):
        self.image_path = large_image_path
        self.gt_path = large_gt_path
        self.mask_path = large_mask_path
        self.w_size = w_size
        self.data_aug = data_aug
        self.aug_mode = aug_mode
        self.dilation = dilation
        
        assert len(large_image_path) == len(large_gt_path) == len(large_mask_path), "Number of input paths must be the same for image, labels and mask"
        image_tiles = []
        gt_tiles = []
        mask_tiles = []
        for i in range(len(large_image_path)):
            image_tiles.append(np.array(generate_tiling_for_data_loader(self.image_path[i], w_size=self.w_size)))
            gt_tiles.append(np.array(generate_tiling_for_data_loader(self.gt_path[i], w_size=self.w_size)))
            mask_tiles.append(np.array(generate_tiling_for_data_loader(self.mask_path[i], w_size=self.w_size)))
        self.image_tiles = np.concatenate(image_tiles)
        self.gt_tiles = np.concatenate(gt_tiles)
        self.mask_tiles = np.concatenate(mask_tiles)
        #self.image_path = np.array(generate_tiling_for_data_loader(self.image_path, w_size=self.w_size))
        #self.gt_path = np.array(generate_tiling_for_data_loader(self.gt_path, w_size=self.w_size))
        #self.mask_path = np.array(generate_tiling_for_data_loader(self.mask_path, w_size=self.w_size))
        print('Window_size: {}, Generate {} image patches, {} gt patches and {} mask patches.'.format(
            w_size, len(self.image_tiles), len(self.gt_tiles), len(self.mask_tiles)))

    def __len__(self):
        return len(self.image_tiles)

    def __getitem__(self, index):
        img = self.image_tiles[index]
        labels = self.gt_tiles[index]
        mask= self.mask_tiles[index]

        if self.dilation:
            struct1 = ndimage.generate_binary_structure(2, 2)
            labels = binary_dilation(labels, structure=struct1).astype(np.uint8)

        if self.data_aug:
            img, labels, mask = transformation(img, labels, mask, self.aug_mode)

        img = img / 255.
        img = np.array(img, dtype=np.float32)

        labels = labels/255.
        labels = labels.astype(np.uint8)

        mask = mask / 255.
        mask = mask.astype(np.uint8)
        
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        labels = torch.from_numpy(np.array([labels])).float()
        mask = torch.from_numpy(mask).float()

        return img, labels, mask, index
'''