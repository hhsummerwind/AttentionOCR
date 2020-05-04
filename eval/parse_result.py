# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: AttentionOCR	
@file: parse_result.py	
@version: v1.0
@time: 2020/5/3 下午10:59
@setting: 
-------------------------------------------------
Description :
工程文件说明： 将所有边界点转换成顺时针。https://www.cnblogs.com/kyokuhuang/p/4250526.html，用端点法测试通过
"""

import json
import numpy as np

# def cal_clock(start, middle, end):
#     ab = middle - start
#     bc = end - middle
#     if np.cross(ab, bc) >= 0:
#         return True
#     else:
#         return False


# def cal_clock(lis):
#     d = 0
#     for i in range(len(lis) - 1):
#         d += -0.5 * (lis[i + 1][1] + lis[i][1]) * (lis[i + 1][0] - lis[i][0])
#     d += -0.5 * (lis[0][1] + lis[-1][1]) * (lis[0][0] - lis[-1][0])
#     if d > 0:
#         return False
#     else:
#         return True


def cal_clock(lis):
    max_x_ind = np.argmax(lis[:,0])
    max_x_point = lis[max_x_ind]
    prev = lis[max_x_ind - 1]
    after_ind = max_x_ind + 1
    if after_ind == len(lis):
        after_ind = 0
    after = lis[after_ind]
    ab = max_x_point - prev
    bc = after - max_x_point

    if np.cross(ab, bc) > 0:
        return True
    else:
        return False


result_file = '/data/models/text_recognition/AttentionOCR/art/task3/end_2_end_result.json'
out_file = '/data/models/text_recognition/AttentionOCR/art/task3/end_2_end_result_right.json'
result = json.load(open(result_file, 'r'))
for k, v in result.items():
    print(k)
    # if k != 'res_3251':
    #     continue
    for item in v:
        points = np.array(item['points'], dtype=np.int)
        # convex_hull = cv2.convexHull(points)
        clockwise = cal_clock(points)
        if not clockwise:
            item['points'] = item['points'][::-1]
json.dump(result, open(out_file, 'w'))
pass