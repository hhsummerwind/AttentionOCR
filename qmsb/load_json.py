# -*- coding: utf-8 -*-
"""
Copyright 2019 TIANJI, Inc. All Rights Reserved.
@author: HuHui
@software: PyCharm	
@project: AttentionOCR	
@file: load_json.py
@version: v1.0
@time: 2019/12/8 下午9:03
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import json
import cv2

# json_path = '/data/datasets/cars/bosch/boxy_labels_valid.json'
json_path = '/data/datasets/cars/bosch/boxy_labels_train.json'
f = open(json_path, 'r')
content = json.load(f)
f.close()
sample_bboxes = content['./bluefox_2016-10-26-12-49-56_bag/1477511371.628746.png']['vehicles']
img_path = '/data/datasets/cars/bosch/bluefox_2016-10-26-12-49-56_bag/1477511371.628746.png'
img = cv2.imread(img_path)
for rec_dict in sample_bboxes:
    for k, v in rec_dict.items():
        if k == 'side':
            continue
        if v is not None:
            img = cv2.rectangle(img, (int(v['y1']), int(v['x1'])), (int(v['y2']), int(v['x2'])),
                                (0, 255, 0), 2)
cv2.imwrite('test.jpg', img)
