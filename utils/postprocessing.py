import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from utils.coco import get_coco_pano_metric


def closing(raster, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    raster = cv2.dilate(raster, kernel, iterations=iterations)
    raster = cv2.erode(raster, kernel, iterations=iterations)
    return raster


def opening(raster, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    raster = cv2.erode(raster, kernel, iterations=iterations)
    raster = cv2.dilate(raster, kernel, iterations=iterations)
    return raster


def remove_white_areas(raster, area_threshold=100):
    # extract connected components from binary raster image
    num_labels, labels, stats, _  = cv2.connectedComponentsWithStats(raster, 8, cv2.CV_32S)
    # remove connected components with size smaller than given area threshold
    for label_idx in range(num_labels):
        label_mask = (labels == label_idx)
        if stats[label_idx, 4] <= area_threshold:
            raster[label_mask] = 0
    return raster 
     

def remove_black_areas(raster, area_threshold=100, white_value=1):
    # invert image
    raster = white_value - raster 
    # extract connected components from binary raster image
    num_labels, labels, stats, _  = cv2.connectedComponentsWithStats(raster, 8, cv2.CV_32S) 
    # remove connected components with size smaller than given area threshold
    for label_idx in range(num_labels):
        label_mask = (labels == label_idx)
        if stats[label_idx, 4] <= area_threshold:
            raster[label_mask] = 0
    # reinvert image
    return white_value - raster 
    



def postprocessing(raster):

    raster = np.uint8(raster)


    # closing to remove noise
    raster = closing(raster, 3, 3)

    # opening to remove noise and fix interrupted roads
    raster = opening(raster, 3, 8)

    
    # connected components
    raster = remove_white_areas(raster, 500)
    raster = remove_black_areas(raster, 500)
    

    return raster