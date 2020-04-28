# -*- coding: utf-8 -*-
"""
Copyright 2019 TIANJI, Inc. All Rights Reserved.
@author: HuHui
@software: PyCharm	
@project: AttentionOCR	
@file: transform_labels.py
@version: v1.0
@time: 2019/12/8 下午4:20
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import pickle
import os

label_path = '/data/models/qmsb/v2/merge_3228_label.pkl'
out_file = os.path.join(os.path.dirname(__file__), 'label.txt')
f_out = open(out_file, 'w')
f = open(label_path, 'rb')
# alphabet = ''.join(pickle.load(f).keys())
label_dict = pickle.load(f)
for i in range(len(label_dict)):
    character = list(label_dict.keys())[list(label_dict.values()).index(i)]
    f_out.write('{} {}\n'.format(i, character))
f_out.write('{} EOS\n'.format(i + 1))
f.close()
f_out.close()
