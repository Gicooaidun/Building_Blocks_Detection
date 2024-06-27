import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from coco_pano_ext_demo import COCO


def get_coco_pano_metric(img_gt_path, img_pred_path):

    img_gt = imread(img_gt_path, as_gray=True)
    img_pred = imread(img_pred_path, as_gray=True)

    T = np.asarray(img_gt == 255)
    P = np.asarray(img_pred == 255)

    PQ, SQ, RQ, score_table = COCO(P, T, ignore_zero=True, output_scores=True)
    
    return PQ, SQ, RQ
