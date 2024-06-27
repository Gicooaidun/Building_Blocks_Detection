import numpy as np
import torch
from torch.utils import data
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
from utils.data_loader_create_tilling import generate_tiling_for_data_loader
from utils.data_aug import transformation

w_size = 512

img_path = "data/train/101-INPUT.jpg"
gt_path = "data/train/101-OUTPUT-GT.png"
mask_path = "data/train/101-INPUT-MASK.png"

image_tiles = np.array(generate_tiling_for_data_loader(img_path, w_size=w_size))
gt_tiles = np.array(generate_tiling_for_data_loader(gt_path, w_size=w_size))
mask_tiles = np.array(generate_tiling_for_data_loader(mask_path, w_size=w_size))

index = 24
img = image_tiles[index]
labels = gt_tiles[index]
mask= mask_tiles[index]

aug_mode = 'ctr+tps'
img, labels, mask = transformation(img, labels, mask, aug_mode)

img = Image.fromarray((img).astype(np.uint8))
labels = Image.fromarray((labels).astype(np.uint8))
mask = Image.fromarray((mask).astype(np.uint8))

img.save(f"data/train/101-INPUT_tile-{index}.jpg")
labels.save(f"data/train/101-INPUT_tile-{index}.png")
mask.save(f"data/train/101-INPUT_tile-{index}.png")