# -*- coding: utf-8 -*-
"""
Copyright 2019 TIANJI, Inc. All Rights Reserved.
@author: HuHui
@software: PyCharm	
@project: AttentionOCR	
@file: test_shengchan10.py
@version: v1.0
@time: 2019/12/16 下午4:26
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""
import pickle
import os
label_path = '/data/datasets/tianji/qmsb/shengchan10/label.txt'
count = 0
all_count = 0
all_names = os.listdir('/data/datasets/tianji/qmsb/shengchan10/qmbd/pass') + \
    os.listdir('/data/datasets/tianji/qmsb/shengchan10/qmbd/unpass')
out_path = '/data/datasets/tianji/qmsb/shengchan10/qmbd/label.pkl'
out_f = open(out_path, 'wb')
label_dict = dict()
f = open(label_path, 'r')
for line in f.readlines():
    contents = line.strip().split('\t')
    img_path = contents[0]
    if img_path not in all_names:
        continue
    else:
        all_count += 1
    name = contents[1]
    if len(name) <= 7:
        count += 1
        label_dict.update({img_path: name})
f.close()
pickle.dump(label_dict, out_f)
print(all_count, count)
out_f.close()