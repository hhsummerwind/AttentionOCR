# -*- coding: utf-8 -*-
"""
Copyright 2019 TIANJI, Inc. All Rights Reserved.
@author: HuHui
@software: PyCharm	
@project: AttentionOCR	
@file: get_character.py
@version: v1.0
@time: 2019/12/5 下午3:12
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import cv2

img_path = '1123076803.jpg'
img = cv2.imread(img_path)
# cv2.imwrite('test.jpg', (img[:, :, 2]>=180)*255)
cv2.imwrite('test_g.jpg', (img[:, :, 1]<20)*255)
# cv2.imwrite('test_b.jpg', img[:, :, 0])