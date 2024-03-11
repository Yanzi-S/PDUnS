import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from skimage.segmentation import find_boundaries
from collections import deque
import numpy_indexed as npi
from utils import minimum_distance, dialate, get_boundary_coordinates, maximum_distance, find_contour_vertices, get_coordinates_in_ring, minimum_distance_two_directions
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

def spiral_points_generation(seed, previous_region, mask, k, start, direction):
    spiral_final = get_coordinates_in_ring(seed[0], k, previous_region, mask, start, direction)
    return spiral_final

def spiral_region_growing(masks, seed, upper_num, classes):
    image_shape = masks.shape
    region = np.zeros(image_shape)
    row, col = seed[0]
    region[row, col] = classes
    k = 1
    start, direction = minimum_distance_two_directions(seed[0], masks)
    while np.count_nonzero(region == classes) < upper_num:
        spiral_points = spiral_points_generation(seed, region, masks, k, start, direction)
        for point in spiral_points:
            x, y = point
            if np.count_nonzero(region == classes) < upper_num:
                region[x, y] = classes
        k = k + 1
    return region


def find_closest_nonzero_coordinates(array1, array2, seed):
    # 找到数组1和数组2中非零元素的坐标
    coords1 = get_boundary_coordinates(array1)
    coords2 = get_boundary_coordinates(array2)
    if np.any(np.array(coords2.shape) == 0) or np.any(np.array(coords1.shape) == 0) or np.array_equal(coords1, coords2):
        aa = 1
    try:
        coords1_tmp = npi.difference(coords1, coords2)
    except Exception as e:
        coords1_tmp = coords1
    try:
        coords2_tmp = npi.difference(coords2, coords1)
    except Exception as e1:
        coords2_tmp = coords2
    if coords2_tmp.size==0:
        coords2_tmp = coords2
    distances = []
    coords_array1 = []
    for i in range(coords2_tmp.shape[0]):
        coords1_tmp1 = np.concatenate((coords1_tmp[coords1_tmp[:, 0] == coords2_tmp[i,0], :], coords1_tmp[coords1_tmp[:, 1] == coords2_tmp[i, 1]]), axis=0)
        condition = []
        right = np.argwhere((coords1_tmp1[:, 0] == coords2_tmp[i, 0]) & (coords1_tmp1[:, 1] > coords2_tmp[i, 1]))
        left = np.argwhere((coords1_tmp1[:, 0] == coords2_tmp[i, 0]) & (coords1_tmp1[:, 1] < coords2_tmp[i, 1]))
        down = np.argwhere((coords1_tmp1[:, 0] > coords2_tmp[i, 0]) & (coords1_tmp1[:, 1] == coords2_tmp[i, 1]))
        up = np.argwhere((coords1_tmp1[:, 0] < coords2_tmp[i, 0]) & (coords1_tmp1[:, 1] == coords2_tmp[i, 1]))
        for ir in range(right.size):
            if array1[coords1_tmp1[right[ir][0]][0],coords1_tmp1[right[ir][0]][1]+1]!=0:
                condition.append(right[ir])
        for il in range(left.size):
            if array1[coords1_tmp1[left[il][0]][0],coords1_tmp1[left[il][0]][1]-1]!=0:
                condition.append(left[il])
        for id in range(down.size):
            if array1[coords1_tmp1[down[id][0]][0]+1,coords1_tmp1[down[id][0]][1]]!=0:
                condition.append(down[id])
        for iu in range(up.size):
            if array1[coords1_tmp1[up[iu][0]][0]-1,coords1_tmp1[up[iu][0]][1]]!=0:
                condition.append(up[iu])
        if len(condition)>0:
            coords1_tmp1=np.delete(coords1_tmp1, condition, axis=0)
        distance = np.linalg.norm(coords2_tmp[i, np.newaxis, :] - coords1_tmp1, axis=-1, ord=1)
        if np.size(distance)==0: # coords2_tmp[i] is at the edge
            continue
        else:
            distances.append(np.min(distance))
        min_col = np.argmin(distance)
        coords_array1.append(coords1_tmp1[min_col, :])

    if len(distances)==0:
        aa=1
    max_count_column_indexes = np.argwhere(distances == np.max(distances)).flatten()
    vertices = find_contour_vertices(array2)
    closest_nin_mask=[]
    closest_vertex=[]
    for i in range(max_count_column_indexes.size):
        ross = max_count_column_indexes[i]
        if np.any(np.all(vertices == coords2_tmp[ross], axis=1)) and not np.all(seed == coords2_tmp[ross], axis=1):
            closest_nin_mask = coords_array1[ross]
            closest_vertex = coords2_tmp[ross]
            break
    if len(closest_nin_mask)==0:
        closest_nin_mask = coords_array1[np.argmax(distances)]
    if len(closest_vertex)==0:
        closest_vertex = coords2_tmp[np.argmax(distances)]

    return np.array(list(closest_nin_mask)), np.array(list(closest_vertex))

def rectangle_size(upper_num):
    s = np.floor(np.sqrt(upper_num)).astype(np.int64)
    if upper_num == s ** 2:
        c = np.arange(1, dtype=np.int64)
        a = s + c
        b = s + c
    elif s ** 2 < upper_num < (s ** 2 + s):
        t = upper_num - s ** 2
        c = np.arange(np.floor(-0.5 + 0.5 * np.sqrt(1 + 4 * s - 4 * t)) + 1, dtype=np.int64)
        a = s + 1 + c
        b = s - c
    elif upper_num == (s ** 2 + s):
        c = np.arange(1, dtype=np.int64)
        a = s + 1 + c
        b = s + c
    else:
        t = upper_num - s ** 2 - s
        c = np.arange(np.floor(np.sqrt(1 + s - t)) + 1, dtype=np.int64)
        a = s + 1 + c
        b = s + 1 - c
    return a, b, s

def rectangle_enumerate(masks, seed, upper_num, classes, window, overlap, a, b):
    row, col = seed[0]
    # determine the location of rectangle (a, b), especially 4 for squares as rotation with degree of 0,90,180,and 270 are the same
    regions = []
    for i in range(a.size):
        if a[i] == b[i]:
            iteration = 1
        else:
            iteration = 2
        dim = np.array([[a[i], b[i]], [b[i], a[i]]])  # two types of size
        for k in range(iteration):
            siz = dim[k]
            regionda = np.zeros_like(masks)
            start = [row-siz[0]+1, col-siz[1]+1]
            end = [row + siz[0] - 1, col + siz[1] - 1]
            if any(element < 0 for element in start):
                start = [max(row-siz[0]+1, 0), max(col-siz[1]+1, 0)]
            if row+siz[0]>masks.shape[0] or col+siz[1]>masks.shape[1]:
                end = [min(row + siz[0] - 1, masks.shape[0]-1), min(col + siz[1] - 1, masks.shape[1]-1)]
            regionda[start[0]:end[0]+1,start[1]:end[1]+1]=1
            region_ok = np.logical_and(regionda != 0, masks != 0).astype(np.uint8)
            contour, _ = cv2.findContours(region_ok, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(contour[0])
            # 寻找最大内接矩形
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 获取最大内接矩形的尺寸
            width = int(rect[1][0]+1)
            height = int(rect[1][1]+1)
            if np.min([width, height])<np.min(siz) or np.max([width, height])<np.max(siz):
                continue
            else:
                for m in range(np.min(box[:, 1]), np.max(box[:, 1]) - siz[0] + 2, 1):
                    for n in range(np.min(box[:, 0]), np.max(box[:, 0]) - siz[1] + 2, 1):
                        region = np.zeros_like(masks)
                        start = [m, n]
                        end = [start[0] + siz[0] - 1, start[1] + siz[1] - 1]
                        if any(element <0 for element in start) or any(element1 >= element2 for element1, element2 in zip(end, masks.shape)):
                            continue
                        region[start[0]:end[0] + 1, start[1]:end[1] + 1] = classes
                        region_tmp = np.zeros_like(masks)
                        if np.count_nonzero(np.logical_and(region != 0, masks != 0)) == np.count_nonzero(region):  # case 1
                            if upper_num == a[i] * b[i]:  # no need to delete
                                region = region
                            else:
                                counts_deleted = a[i] * b[i] - upper_num
                                start_tmp = np.array(start) - 2 * window + overlap
                                end_tmp = np.array(end) + 2 * window - overlap
                                start_tmp[start_tmp<0]=0
                                end_tmp1=np.minimum(end_tmp, masks.shape)
                                region_tmp[start_tmp[0]:end_tmp1[0]+1, start_tmp[1]:end_tmp1[1]+1] = classes  # extended regions
                                if np.count_nonzero(np.logical_and(region_tmp != 0, masks != 0)) == np.count_nonzero(region_tmp):  # case 1.1
                                    region[start[0], start[1]:start[1] + counts_deleted] = 0  # default: delete from the upper left
                                else:  # case 1.2
                                    intersection = np.logical_and(region_tmp != 0, masks != 0)
                                    # search a square in region closest to the boundary of intersection
                                    _, closest_coords = find_closest_nonzero_coordinates(intersection, region, seed)
                                    x_dif, y_dif = np.abs(closest_coords - start)
                                    if x_dif == 0 or x_dif == (end[0] - start[0]):  # the square locates at x-axis
                                        if y_dif <= np.abs(end[1] - closest_coords[1]):  # left
                                            if (y_dif + 1) >= counts_deleted:
                                                region[closest_coords[0], closest_coords[1] - counts_deleted + 1:closest_coords[1] + 1] = 0
                                            else:
                                                region[closest_coords[0], start[1]:start[1] + counts_deleted] = 0
                                        else:  # right
                                            if (siz[1] - y_dif) >= counts_deleted:
                                                region[closest_coords[0], closest_coords[1]:closest_coords[1] + counts_deleted] = 0
                                            else:
                                                region[closest_coords[0], end[1] + 1 - counts_deleted:end[1] + 1] = 0
                                    elif y_dif == 0 or y_dif == (end[1] - start[1]):  # the square locates at y-axis
                                        if x_dif <= np.abs(end[0] - closest_coords[0]):  # top
                                            if (x_dif + 1) >= counts_deleted:
                                                region[closest_coords[0] - counts_deleted + 1:closest_coords[0] + 1, closest_coords[1]] = 0
                                            else:
                                                region[start[0]: start[0] + counts_deleted, closest_coords[1]] = 0
                                        else:  # bottom
                                            if (siz[0] - x_dif) >= (a[i] * b[i] - upper_num):
                                                region[closest_coords[0]:closest_coords[0] + counts_deleted, closest_coords[1]] = 0
                                            else:
                                                region[end[0] + 1 - counts_deleted:end[0] + 1, closest_coords[1]] = 0
                        elif np.count_nonzero(np.logical_and(region != 0, masks != 0)) >= upper_num:  # case 2
                            square_nin_mask = np.logical_and(np.logical_and(region != 0, masks != 0) == 0, region != 0)  # squares in region but not in mask
                            nonzero_indices = np.transpose(np.nonzero(square_nin_mask))
                            region[square_nin_mask != 0] = 0
                            if (nonzero_indices.shape[0] + upper_num) == a[i] * b[i]:
                                region=region
                            else:
                                counts_deleted = a[i] * b[i] - upper_num - nonzero_indices.shape[0]
                                region_tmp = np.zeros_like(masks)
                                start_tmp = np.array(start) - 2 * window + overlap
                                end_tmp = np.array(end) + 2 * window - overlap
                                start_tmp[start_tmp<0]=0
                                end_tmp1=np.minimum(end_tmp, masks.shape)
                                region_tmp[start_tmp[0]:end_tmp1[0]+1, start_tmp[1]:end_tmp1[1]+1] = classes  #
                                intersection = np.logical_and(region_tmp != 0, masks != 0)
                                closest_nin_mask, closest_vertex = find_closest_nonzero_coordinates(intersection, region, seed)
                                closest_coords = closest_vertex
                                x_dif, y_dif = np.abs(closest_coords - start)
                                if x_dif == 0 or x_dif == (end[0] - start[0]):  # the square locates at x-axis
                                    if y_dif <= np.abs(end[1] - closest_coords[1]):  # left
                                        if (y_dif + 1) >= counts_deleted:
                                            region[closest_coords[0], closest_coords[1] - counts_deleted + 1:closest_coords[1] + 1] = 0
                                        else:
                                            region[closest_coords[0], start[1]:start[1] + counts_deleted] = 0
                                    else:  # right
                                        if (siz[1] - y_dif) >= counts_deleted:
                                            region[closest_coords[0], closest_coords[1]:closest_coords[1] + counts_deleted] = 0
                                        else:
                                            region[closest_coords[0], end[1] + 1 - counts_deleted:end[1] + 1] = 0
                                elif y_dif == 0 or y_dif == (end[1] - start[1]):  # the square locates at y-axis
                                    if x_dif <= np.abs(end[0] - closest_coords[0]):  # top
                                        if (x_dif + 1) >= counts_deleted:
                                            region[closest_coords[0] - counts_deleted + 1:closest_coords[0] + 1, closest_coords[1]] = 0
                                        else:
                                            region[start[0]: start[0] + counts_deleted, closest_coords[1]] = 0
                                    else:  # bottom
                                        if (siz[0] - x_dif) >= (a[i] * b[i] - upper_num):
                                            region[closest_coords[0]:closest_coords[0] + counts_deleted, closest_coords[1]] = 0
                                        else:
                                            region[end[0] + 1 - counts_deleted:end[0] + 1,
                                            closest_coords[1]] = 0
                        else:
                            region[region!=0]=0
                    if np.count_nonzero(region) != 0:
                        regions.append(region)

    return regions


def PMnOS(label, ratio, window, overlap, seeds=None):
    out_img = np.zeros_like(label)
    if seeds is None:
        seeds = []
    k = 0
    for classes in np.unique(label):
        if classes != 0:
            masks = np.where(label == classes, 1, 0)
            masks = np.array(masks * 255, np.uint8)
            num_labels, connectPlane, stats, centroids = cv2.connectedComponentsWithStats(masks, connectivity=8)
            out_img_classes = np.zeros_like(masks)
            for i in range(1, num_labels):
                coords = np.column_stack(np.where(connectPlane == i))
                if len(seeds) <= k:
                    seed = coords[np.random.randint(0, coords.shape[0] - 1, size=1)]
                    seeds.extend(seed)
                else:
                        seed = np.reshape(seeds[k], (1, -1))
                k = k + 1
                # obtain the width and height of satisfying rectangles
                upper_num = np.ceil(coords.shape[0] * ratio).astype(np.int64)
                a, b, s = rectangle_size(upper_num)
                regions = rectangle_enumerate(np.where(connectPlane == i, 1, 0), seed, upper_num, classes,                                      window // 2, overlap, a, b)
                if len(regions) == 0:
                    regions.append(spiral_region_growing(np.where(connectPlane == i, 1, 0), seed, upper_num, classes))
                kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
                numTest = []
                for q in range(len(regions)):
                    masks1 = regions[q]
                    masks1 = masks1.astype(np.uint8)
                    dilated_image = cv2.dilate(masks1, kernel, iterations=(2 * window//2 - overlap))
                    result_image = np.logical_or(masks1, dilated_image).astype(np.uint8)
                    out_ima_classes = np.logical_xor(label == classes, np.logical_and(result_image == 1, label == classes) == True)
                    numTest.append(np.count_nonzero(out_ima_classes.astype(np.int64)))
                min_index = numTest.index(max(numTest))

                out_img_classes = out_img_classes + regions[min_index]

            out_img = out_img + out_img_classes
    return out_img, seeds
