import RegionGrowth
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
from collections import deque

def region_growing(image, seed, upperNum, classes):
    height, width = image.shape
    segmented = np.zeros_like(image)
    stack = [row.tolist() for row in seed]
    my_deque = deque(stack)
    while np.count_nonzero(segmented == classes) < upperNum:
        current_point = my_deque.popleft()
        row, col = current_point
        if 0 <= row < height and 0 <= col < width:
            if segmented[row, col] == 0:
                segmented[row, col] = classes
            neighbors = [
            [row - 1, col - 1], [row - 1, col], [row - 1, col + 1],
            [row, col - 1], [row, col + 1],
            [row + 1, col - 1], [row + 1, col], [row + 1, col + 1]
            ]
            k=0
            while np.count_nonzero(segmented == classes) < upperNum and k<8:
                rowTmp, colTmp = neighbors[k]
                if image[rowTmp,colTmp] == image[row, col] and segmented[rowTmp,colTmp]==0:
                    segmented[rowTmp,colTmp] = classes
                    my_deque.append(neighbors[k])
                k=k+1
    return segmented

def CRS(label, ratio, seeds=None):
    out_img=np.zeros_like(label)
    if seeds is None:
        seeds = []
    k = 0
    for classes in np.unique(label):
        if classes !=0:
            masks = np.where(label==(classes), 1, 0)
            masks = np.array(masks * 255, np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masks, connectivity=8)
            out_img_classes=np.zeros_like(masks)
            for i in range(1,num_labels):
                coords = np.column_stack(np.where(labels == i))
                if len(seeds)<=k:
                    seed = coords[np.random.randint(0,coords.shape[0]-1,size=1)]
                    seeds.extend(seed)
                else:
                    seed = np.reshape(seeds[k],(1,-1))
                k = k+1
                out_img_classes = out_img_classes + region_growing(masks, seed, np.ceil(coords.shape[0]*ratio), classes)

            out_img = out_img + out_img_classes
    return out_img.astype(np.int64), seeds