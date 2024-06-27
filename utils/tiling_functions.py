import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.util import view_as_windows
from sklearn.metrics import accuracy_score, f1_score


def generate_tiling(image_path, w_size):
    '''
    Generate image tiles.
    :param image_path: string path of image
    :param w_size: int size of tiles (window size)
    :return: np array of tiles
    '''

    win_size = w_size
    pad_px = win_size // 2

    # Read image
    in_img = np.array(Image.open(image_path))

    img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'constant')
    tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px) # function to get image tilings -> step equals patch size/2
    
    tiles_lst = []
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            tt = tiles[row, col, 0, ...].copy()
            tiles_lst.append(tt)
    tiles_array = np.concatenate(tiles_lst)

    # You must reshape the tiles_array into (batch_size, width, height, 3)
    tiles_array = tiles_array.reshape(int(tiles_array.shape[0]/w_size), w_size, w_size, 3)
    return tiles_array



def reconstruct_from_tiles(image_tiles, tile_size, step_size, image_size_2d, image_dtype):
    '''
    Recunstruct image from overlapping tiles.
    :param image_tiles: numpy array with tiles of shape (n_tiles, tile_size, tile_size)
    :param tile_size: int tile size
    :param step_size: int step size (i.e. overlap of tiles). Normally tile_size/2
    :param image_size_2d: tuple with size of the original image
    :param image_dtype: data type of target image
    :return: target image
    '''
    i_h, i_w = np.array(image_size_2d[:2]) + (tile_size, tile_size)
    p_h = p_w = tile_size
    if len(image_tiles.shape) == 4:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2, 3), dtype=image_dtype)
        print('used this against expectation')
    else:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2), dtype=image_dtype)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1
    expected_tiles = numrows * numcols
    if len(image_tiles) != expected_tiles:
        raise ValueError(f"Expected {expected_tiles} tiles, got {len(image_tiles)}")

    tile_offset = step_size//2
    tile_inner = p_h-step_size
    for row in range(numrows):
        for col in range(numcols):
            tt = image_tiles[row*numcols+col]
            tt_roi = tt[tile_offset:-tile_offset,tile_offset:-tile_offset]
            img[row*step_size:row*step_size+tile_inner,
                col*step_size:col*step_size+tile_inner] = tt_roi # +1?? 
    return img[step_size//2:-(tile_size+step_size//2),step_size//2:-(tile_size+step_size//2),...]