import csv
import os

import cv2
import gin
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from shapely.geometry import Polygon
from tqdm import tqdm


def get_images(data_path):
    """
    find image files in test data path
    :return: list of files found
    """

    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Found {} images'.format(len(files)))
    return files


def load_annoataion(txt_file_path):
    """
    load annotation from the text file
    "x1, y1, x2, y2, x3, y3, x4, y4, transcription"

    :param txt_file_path:
    :return:
    """

    text_polys = []
    text_tags = []  # mask used in training, to ignore some text annotated by ###
    # if not os.path.exists(p):
    #     return np.array(text_polys, dtype=np.float32)

    try:
        f = open(txt_file_path, 'r')
    except FileNotFoundError:
        raise FileNotFoundError
    else:
        with f:
            reader = csv.reader(f)
            for line in reader:
                label = line[-1]
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                # In ICDAR 2015 dataset unreadable text regions are annotated with either * or #
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)

            return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def polygon_area(poly):
    """
    compute area of a polygon
    :param poly:
    :return:
    """

    """
    https://math.stackexchange.com/questions/1259094/coordinate-geometry-area-of-a-quadrilateral
    https://en.wikipedia.org/wiki/Shoelace_formula
     |0 |1 |
    -|--|--|
    0|x1|y1|
    1|x2|y2|
    2|x3|y3|
    3|x4|y4|

    x1,y1                          x2,y2
    ----------------------------------
    |                                |
    |                                |
    |                                |
    ----------------------------------
    x4,y4                         x3,y3
    """

    edge = [
        #  (x2 - x1) * (y2 + y1)
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        #  (x3 - x2) * (y3 + y2)
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        #  (x4 - x3) * (y4 + y3)
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        #  (x1 - x4) * (y1 + y4)
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polys, tags, height_weight_tuple):
    """
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    """

    (h, w) = height_weight_tuple

    if polys.shape[0] == 0:
        return polys

    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []

    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            print('invalid poly')
            continue
        if p_area > 0:  # TODO what maths is involved?
            print('poly in wrong direction')
            # rows are inter changed
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, min_crop_side_ratio, crop_background=False, max_tries=50):
    """
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param min_crop_side_ratio
    :param crop_background:
    :param max_tries:
    :return:
    """
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) & (polys[:, :, 1] >= ymin) & (
                    polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def shrink_poly(poly, r):
    """
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    """

    """

      p0                         p1
       --------------------------
      |                          |
      |                          |
      |                          |
       --------------------------  
      p3                         p2 
    """
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        # p0, p1               p1_y - p0_y    p1_x - p0_x
        # https://en.wikipedia.org/wiki/Atan2
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        # p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)  # TODO find some simple example
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    """
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    """

    """

      p0                         p1
       --------------------------
      |                          |
      |                          |
      |                          |
       --------------------------  
      p3                         p2 
    """

    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))

    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            # p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            # p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            # p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            # p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def generate_rbox(im_size, polys, tags, min_text_size):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during training, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)

    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        # ------------------------------------------------------------------------------------------------
        # Section 3.3.1 of EAST paper
        # Clockwise distance between two points and anti clockwise distance between two points

        """

          0                          1
           --------------------------
          |                          |
          |                          |
          |                          |
           --------------------------  
          3                          2 
        """

        reference_length = [None, None, None, None]
        for i in range(4):
            # find the shorted edge distances(vertices1,vertices2)
            # (0,1) | (1,2) | (2,3) | (3,0) <- (i+1) % 4
            # (0,3) | (1,0) | (2,1) | (3,2) <- (i-1) % 4
            reference_length[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                                      np.linalg.norm(poly[i] - poly[(i - 1) % 4]))

        # ------------------------------------------------------------------------------------------------

        # score map
        shrinked_poly = shrink_poly(poly.copy(), reference_length).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)  # TODO ?

        # ------------------------------------------------------------------------------------------------

        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # if the poly is too small or unreadable text is found, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))

        if min(poly_h, poly_w) < min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # ------------------------------------------------------------------------------------------------

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

        # if geometry == 'RBOX':
        # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices

        # We first generate a rotated rectangle that covers the region with minimal area.
        # Then for each pixel which has positive score, we calculate its distances
        # to  the  4  boundaries  of  the  text  box, and put  them to the 4 channels of RBOX ground truth.

        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]

            # [p0_x, p1_x], [p0_y, p0_y]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            # [p0_x, p3_x], [p0_y, p3_y]
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            # [p1_x, p2_x], [p1_y, p2_y]
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])

            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # 平行线经过p2 - parallel lines through p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3 - after p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]

            # --------------------------------------------------------------------------------------

            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)

            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]

            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)

            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])

            # --------------------------------------------------------------------------------------

            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)

            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]

            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])

        # --------------------------------------------------------------------------------------------------

        # print("Lenght of fitted_parallelograms : ", len(fitted_parallelograms))

        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)

        # sort the polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectangle = rectangle_from_parallelogram(parallelogram)
        rectangle, rotate_angle = sort_rectangle(rectangle)

        p0_rect, p1_rect, p2_rect, p3_rect = rectangle

        # --------------------------------------------------------------------------------------------------

        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle

    return score_map, geo_map, training_mask


def print_shape(mat, name="name"):
    print(name, mat.shape)


def image_2_data(image_file_path,
                 geometry,
                 min_text_size,
                 min_crop_side_ratio,
                 input_size=512,
                 background_ratio=3. / 8,
                 random_scale=np.array([0.5, 1, 2.0]),  # , 3.0]),
                 vis=False):
    images = []
    image_fns = []
    score_maps = []
    geo_maps = []
    training_masks = []
    found_text_file = False
    try:
        im = cv2.imread(image_file_path)
        # print image_file_path
        h, w, _ = im.shape

        # repalce extenstion
        # img_1.png -> img_1.txt
        txt_file_name = image_file_path.replace(os.path.basename(image_file_path).split('.')[1], 'txt')

        # TODO clean this out!
        try:
            # 2019 dataset : img_1.png -> img_1.txt
            text_polys, text_tags = load_annoataion(txt_file_name)
            if os.path.exists(txt_file_name):
                found_text_file = True
        except:
            # 2015 dataset : #img_1.txt -> gt_img_01.txt
            # img_1.png -> img_1.txt
            txt_file_name = image_file_path.replace(os.path.basename(image_file_path).split('.')[1], 'txt')
            # img_1.txt -> gt_img_01.txt
            txt_file_name = txt_file_name.replace(os.path.basename(txt_file_name).split('.')[0], 'gt_' +
                                                  os.path.basename(txt_file_name).split('.')[0])
            try:
                text_polys, text_tags = load_annoataion(txt_file_path=txt_file_name)
                if os.path.exists(txt_file_name):
                    found_text_file = True
            except:
                found_text_file = False

        if not found_text_file:
            return

        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

        # random scale this image
        rd_scale = np.random.choice(random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale

        # random crop a area from image
        # if np.random.rand() < background_ratio:
        #     print("Dummy score map and geo map...")
        #     # crop background
        #     im, text_polys, text_tags = crop_area(im, text_polys, text_tags,
        #                                           min_crop_side_ratio=min_crop_side_ratio,
        #                                           crop_background=True)
        #     if text_polys.shape[0] > 0:
        #         # cannot find background
        #         return
        #
        #     # pad and resize image
        #     new_h, new_w, _ = im.shape
        #     max_h_w_i = np.max([new_h, new_w, input_size])
        #     im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        #     im_padded[:new_h, :new_w, :] = im.copy()
        #     im = cv2.resize(im_padded, dsize=(input_size, input_size))
        #
        #     score_map = np.zeros((input_size, input_size), dtype=np.uint8)
        #
        #     geo_map_channels = 5 if geometry == 'RBOX' else 8
        #     geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
        #
        #     training_mask = np.ones((input_size, input_size), dtype=np.uint8)
        # else:
        im, text_polys, text_tags = crop_area(im, text_polys, text_tags,
                                              min_crop_side_ratio=min_crop_side_ratio,
                                              crop_background=False)
        if text_polys.shape[0] == 0:
            return
        h, w, _ = im.shape

        # pad the image to the training input size or the longer side of image
        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = im_padded

        # resize the image to input size
        new_h, new_w, _ = im.shape
        resize_h = input_size
        resize_w = input_size
        im = cv2.resize(im, dsize=(resize_w, resize_h))

        resize_ratio_3_x = resize_w / float(new_w)
        resize_ratio_3_y = resize_h / float(new_h)

        text_polys[:, :, 0] *= resize_ratio_3_x
        text_polys[:, :, 1] *= resize_ratio_3_y
        new_h, new_w, _ = im.shape

        score_map, geo_map, training_mask = generate_rbox((new_h, new_w),
                                                          text_polys,
                                                          text_tags,
                                                          min_text_size=min_text_size)

        if vis:
            plt.imshow(im)
            print_shape(im, "im")

            fig, axs = plt.subplots(3, 2, figsize=(20, 30))

            axs[0, 0].imshow(im[:, :, ::-1])
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            for poly in text_polys:
                poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                axs[0, 0].add_artist(Patches.Polygon(poly,
                                                     facecolor='none',
                                                     edgecolor='green',
                                                     linewidth=2,
                                                     linestyle='-',
                                                     fill=True))
                axs[0, 0].text(poly[0, 0],
                               poly[0, 1],
                               '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
            axs[0, 0].set_title("text RBOX")

            axs[0, 1].imshow(score_map[::, ::])
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 1].set_title("score_map")

            axs[1, 0].imshow(geo_map[::, ::, 0])
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 0].set_title("geo_map0")

            axs[1, 1].imshow(geo_map[::, ::, 1])
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            axs[1, 1].set_title("geo_map1")

            axs[2, 0].imshow(geo_map[::, ::, 2])
            axs[2, 0].set_xticks([])
            axs[2, 0].set_yticks([])
            axs[2, 0].set_title("geo_map2")

            axs[2, 1].imshow(training_mask[::, ::])
            axs[2, 1].set_xticks([])
            axs[2, 1].set_yticks([])
            axs[2, 0].set_title("training_mask")

            plt.tight_layout()
            plt.show()
            plt.close()

        # print_shape(im, "image")
        # print_shape(score_map, "score_map")
        # print_shape(geo_map, "geo_map")
        # print_shape(training_mask, "training_mask")

        image = im[:, :, ::-1].astype(np.float32)
        # scale rest of the ata by 4 i.e sample 1 for every 4 pixels
        score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
        geo_map = geo_map[::4, ::4, :].astype(np.float32)
        training_mask = training_mask[::4, ::4, np.newaxis].astype(np.float32)

        # print_shape(image, "image")
        # print_shape(score_map, "score_map")
        # print_shape(geo_map, "geo_map")
        # print_shape(training_mask, "training_mask")
        return image, score_map, geo_map, training_mask

    except Exception as e:
        import traceback
        traceback.print_exc()


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ======================================================================================================================
