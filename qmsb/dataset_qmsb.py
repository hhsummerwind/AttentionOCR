#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

import cv2
from matplotlib import pyplot as plt
import json
import codecs
import pdb

from PIL import Image, ImageDraw, ImageFont 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from qmsb import config_qmsb as cfg

max_len = cfg.seq_len + 1
# base_dir = cfg.base_dir
font_path = cfg.font_path

# dataset_path = {  'art': os.path.join(base_dir, 'art/train_task2_images'),}
#                   # 'rects': os.path.join(base_dir, 'rects/img'),
#                   # 'lsvt': os.path.join(base_dir, 'lsvt/train'),
#                   # 'icdar2017rctw': os.path.join(base_dir, 'icdar2017rctw/train'), }
#
# lsvt_annotation = os.path.join(base_dir, 'lsvt/train_full_labels.json')
# art_annotation = os.path.join(base_dir, 'art/train_task2_labels.json')



def visualization(image_path, points, label, vis_color = (255,255,255)):
    """
    Visualize groundtruth label to image.
    """
    points = np.asarray(points, dtype=np.int32)
    points = np.reshape(points, [-1,2])
    image = cv2.imread(image_path)
    cv2.polylines(image, [points], 1, (0,255,0), 2)
    image = Image.fromarray(image)
    FONT = ImageFont.truetype(font_path, 20, encoding='utf-8')   
    DRAW = ImageDraw.Draw(image)  
    
    DRAW.text(points[0], label, vis_color, font=FONT)
    return np.array(image)

def strQ2B(uchar):
    """
    Convert full-width character to half-width character.
    """
    inside_code = ord(uchar)
    if inside_code == 12288:
        inside_code = 32
    elif (inside_code >= 65281 and inside_code <= 65374):
        inside_code -= 65248
    return chr(inside_code)

def preprocess(string):
    """
    Groundtruth label preprocess function.
    """
    # string = [strQ2B(ch) for ch in string.strip()]
    # return ''.join(string)  
    return string  


class Dataset(object):
    """
    Base class for text dataset preprocess.
    """
    def __init__(self, name='base', max_len=max_len, label_dict=cfg.reverse_label_dict): # label_dict  label_dict_with_rects 5434+1
        # self.data_path = dataset_path[name]
        # print(self.data_path)
        self.label_dict = label_dict
        self.max_len = max_len
        # self.base_dir = base_dir
        self.filenames = []
        self.labels = []
        # self.masks = []
        # self.bboxes = []
        # self.points = []


class QMSB(Dataset):
    """
    qmsb dataset of Tianji
    """

    def __init__(self, name='qmsb'):
        super(QMSB, self).__init__(name=name)

    def load_data(self, annotation_file):
        # pdb.set_trace()
        count = 0
        with open(annotation_file) as f:
            # json_data = json.load(f)

            # for filename in os.listdir(self.data_path):
            for line in f.readlines():
                contents = line.strip().split('\t')
                # img_name = os.path.join(self.data_path, filename)
                img_name = contents[0]
                # image = cv2.imread(img_name)
                # image_height, image_width = image.shape[:2]

                # anno_data = dict()#json_data[filename[:-4]][0]
                # print(len(json_data[filename[:-4]]))
                # illegibility = anno_data['illegibility']

                # if illegibility:
                #     continue

                # polygon = anno_data['points']
                transcripts = contents[1]#anno_data['transcription']
                # languages = anno_data['language']

                if len(transcripts) > self.max_len - 1:
                    # print(transcripts)
                    # count = count + 1
                    continue

                # transcripts = preprocess(transcripts)

                skip = False
                for char in transcripts:
                    if char not in self.label_dict.keys():
                        skip = True

                if skip:
                    # print(transcripts)
                    count = count + 1
                    continue

                # print(polygon, transcripts)

                seq_label = []
                for char in transcripts:
                    seq_label.append(self.label_dict[char])  # .decode('utf-8')
                seq_label.append(self.label_dict['EOS'])

                non_zero_count = len(seq_label)
                seq_label = seq_label + [self.label_dict['EOS']] * (self.max_len - non_zero_count)
                # mask = [1] * (non_zero_count) + [0] * (self.max_len - non_zero_count)
                #
                # points_x = [point[0] for point in polygon]
                # points_y = [point[1] for point in polygon]
                # bbox = [np.amin(points_y), np.amin(points_x), np.amax(points_y),
                #         np.amax(points_x)]  # ymin, xmin, ymax, xmax
                # bbox = [int(item) for item in bbox]
                #
                # bbox_w, bbox_h = bbox[3] - bbox[1], bbox[2] - bbox[0]
                #
                # if bbox_w < 8 or bbox_h < 8:
                #     continue

                self.filenames.append(img_name)
                self.labels.append(seq_label)
                # self.masks.append(mask)
                # self.bboxes.append(bbox)
                # self.points.append(polygon)


if __name__=='__main__':
    # LSVT = LSVT()
    # LSVT.load_data()
    # print(len(LSVT.filenames))

    qmsb_data = QMSB()
    qmsb_data.load_data(cfg.label_path)
    print(len(qmsb_data.filenames))

    # ReCTS = ReCTS()
    # ReCTS.load_data()
    # print(len(ReCTS.filenames))
    
    filenames = qmsb_data.filenames#LSVT.filenames + ART.filenames + ReCTS.filenames
    labels = qmsb_data.labels#LSVT.labels + ART.labels + ReCTS.labels
    # masks = ART.masks#LSVT.masks + ART.masks + ReCTS.masks
    # bboxes = ART.bboxes#LSVT.bboxes + ART.bboxes + ReCTS.bboxes
    # points = ART.points#LSVT.points + ART.points + ReCTS.points

    from sklearn.utils import shuffle
    filenames, labels = shuffle(filenames, labels,random_state=999)
    print(len(filenames))

    dataset = {"filenames":filenames, "labels":labels}
    np.save(cfg.dataset_name, dataset)

    
    


    
    
    
    