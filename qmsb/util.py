# -*- coding: utf-8 -*-
"""
Copyright 2019 TIANJI, Inc. All Rights Reserved.
@author: HuHui
@software: PyCharm	
@project: AttentionOCR	
@file: util.py
@version: v1.0
@time: 2019/12/8 下午5:34
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import numpy as np

def find_continous_zeros(a):
    a = np.concatenate(([0], a == 0, [0]))
    absdiff = np.abs(np.diff(a))
    indexes = np.where(absdiff == 1)[0]
    if len(indexes) % 2 != 0:
        indexes = np.append(indexes, a.shape[0])
    ranges = indexes.reshape(-1, 2)
    return ranges


def find_blank_side(img, th_side):
    h, w = img.shape
    sum_rows = np.sum(img, 0)
    ind_rows = sum_rows != 255 * h
    row_ranges = find_continous_zeros(ind_rows)
    sum_cols = np.sum(img, 1)
    ind_cols = sum_cols != 255 * w
    col_ranges = find_continous_zeros(ind_cols)
    if len(row_ranges) == 0:
        ul_x = 0
    elif row_ranges[0][0] != 0:
        ul_x = 0
    elif row_ranges[0][1] - row_ranges[0][0] < th_side:
        ul_x = 0
    else:
        ul_x = row_ranges[0][1] - row_ranges[0][0] - th_side
    if len(row_ranges) == 0:
        lr_x = w
    elif row_ranges[-1][1] != w:
        lr_x = w
    elif row_ranges[-1][1] - row_ranges[-1][0] < th_side:
        lr_x = w
    else:
        lr_x = row_ranges[-1][0] + th_side
    if len(col_ranges) == 0:
        ul_y = 0
    elif col_ranges[0][0] != 0:
        ul_y = 0
    elif col_ranges[0][1] - col_ranges[0][0] < th_side:
        ul_y = 0
    else:
        ul_y = col_ranges[0][1] - col_ranges[0][0] - th_side
    if len(col_ranges) == 0:
        lr_y = h
    elif col_ranges[-1][1] != h:
        lr_y = h
    elif col_ranges[-1][1] - col_ranges[-1][0] < th_side:
        lr_y = h
    else:
        lr_y = col_ranges[-1][0] + th_side
    out_img = img[ul_y:lr_y, ul_x:lr_x]
    return out_img

