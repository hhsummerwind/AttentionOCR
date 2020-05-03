# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: AttentionOCR	
@file: eval_hh.py	
@version: v1.0
@time: 2020/5/2 下午5:56
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

# from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
# from tensorpack.tfutils import SmartInit, get_tf_version_tuple
import sys
sys.path.append('/projects/open_sources/text_recognition/AttentionOCR')
import tensorflow as tf
import numpy as np
import argparse
import time
import cv2
from skimage import draw
import copy
import pdb
import os
import json
import glob

from eval.text_detection import TextDetection
from eval.text_recognition import TextRecognition
# from model.tensorpack_model import *
# from common import polygons_to_mask

use_gpu = True
seq_len = 32
image_size = 299


def init_ocr_model():
    detection_pb = '/data/models/text_recognition/AttentionOCR/checkpoint/ICDAR_0.7.pb' # './checkpoint/ICDAR_0.7.pb'
    # recognition_checkpoint='/data/zhangjinjin/icdar2019/LSVT/full/recognition/checkpoint_3x_single_gpu/OCR-443861'
    # recognition_pb = './checkpoint/text_recognition_5435.pb' #
    recognition_pb = '/data/models/text_recognition/AttentionOCR/checkpoint/text_recognition.pb'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    if use_gpu:
        with tf.device('/gpu:0'):
            tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                       allow_soft_placement=True)

            detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
            recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
    else:
        with tf.device('/cpu:0'):
            tf_config = tf.ConfigProto(allow_soft_placement=True)

            detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
            recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)

    label_dict = np.load('/data/models/text_recognition/AttentionOCR/reverse_label_dict_with_rects.npy', allow_pickle=True)[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
    return detection_model, recognition_model, label_dict


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def preprocess(image, points, size):
    """
    Preprocess for test.
    Args:
        image: test image
        points: text polygon
        size: test image size
    """
    height, width = image.shape[:2]
    mask = polygons_to_mask([np.asarray(points, np.float32)], height, width)
    x, y, w, h = cv2.boundingRect(mask)
    mask = np.expand_dims(np.float32(mask), axis=-1)
    image = image * mask
    image = image[y:y + h, x:x + w, :]

    new_height, new_width = (size, int(w * size / h)) if h > w else (int(h * size / w), size)
    image = cv2.resize(image, (new_width, new_height))

    if new_height > new_width:
        padding_top, padding_down = 0, 0
        padding_left = (size - new_width) // 2
        padding_right = size - padding_left - new_width
    else:
        padding_left, padding_right = 0, 0
        padding_top = (size - new_height) // 2
        padding_down = size - padding_top - new_height

    image = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right,
                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image = image / 255.
    return image


def cal_sim(str1, str2):
    """
    Normalized Edit Distance metric (1-N.E.D specifically)
    """
    m = len(str1) + 1
    n = len(str2) + 1
    matrix = np.zeros((m, n))
    for i in range(m):
        matrix[i][0] = i

    for j in range(n):
        matrix[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(matrix[i - 1][j - 1], min(matrix[i][j - 1], matrix[i - 1][j])) + 1

    lev = matrix[m - 1][n - 1]
    if (max(m - 1, n - 1)) == 0:
        sim = 1.0
    else:
        sim = 1.0 - lev / (max(m - 1, n - 1))
    return sim


def label2str(preds, probs, label_dict, eos='EOS'):
    """
    Predicted sequence to string.
    """
    results = []
    for idx in preds:
        if label_dict[idx] == eos:
            break
        results.append(label_dict[idx])

    probabilities = probs[:min(len(results) + 1, seq_len + 1)]
    return ''.join(results), probabilities


def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox


def eval_crop(detection_model, recognition_model, args, filenames, polygons, labels, label_dict, out_f):
    Normalized_ED = 0.
    total_num = 0
    total_time = 0

    for i, (filename, points, label) in enumerate(zip(filenames, polygons, labels)):
        image = cv2.imread(filename)
        vis_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # r_boxes, polygons, scores = detection_model.predict(image)

        image = preprocess(image, points, image_size)

        before = time.time()
        # preds, probs = predictor(np.expand_dims(image, axis=0), np.ones([1, cfg.seq_len + 1], np.int32), False, 1.)
        # pdb.set_trace()
        preds, probs = recognition_model.predict(np.expand_dims(image, axis=0), label_dict, EOS='EOS')
        # print(preds, probs)
        after = time.time()

        total_time += after - before
        # preds, probs = label2str(preds[0], probs[0], label_dict)
        probs = probs[:min(len(preds) + 1, seq_len + 1)]

        sim = cal_sim(preds, label)
        # print(label)
        out_f.write('image {}: {}, label = {}, prediction = {}, probs = {}, sim = {}\n'.format(i, filename, label,
                                                                                               preds, probs, sim))

        total_num += 1
        Normalized_ED += sim

    print("total_num: %d, 1-N.E.D: %.4f, average time: %.4f" % (
    total_num, Normalized_ED / total_num, total_time / total_num))


def test_intigrate(detection_model, recognition_model, filename, label_dict):
    key = os.path.basename(os.path.splitext(filename)[0])
    result_dict = {key: []}
    image = cv2.imread(filename)
    vis_image = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_boxes, polygons, scores = detection_model.predict(image)
    # pdb.set_trace()
    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height >= width:
            scale = test_size / height
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                              right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.
        else:
            scale = test_size / width
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                              right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.

        image_padded = np.expand_dims(image_padded, 0)

        preds, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
        probs = probs[:min(len(preds) + 1, seq_len + 1)]
        result_dict[key].append({"transcription": ''.join(preds).encode('unicode_escape'), "points": polygon.tolist(),
                                 "confidence": score})
    return result_dict

        # probs = probs[:min(len(preds) + 1, seq_len + 1)]

        # out_f.write('image {}: {}, label = {}, prediction = {}, probs = {}, sim = {}\n'.format(i, filename, label,
        #                                                                                        preds, probs, sim))

        # total_num += 1
        # Normalized_ED += sim

    # print("total_num: %d, 1-N.E.D: %.4f, average time: %.4f" % (
    # total_num, Normalized_ED / total_num, total_time / total_num))


def test_2019icdar_art_train_task2():
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model',
                        default='./checkpoint/model-10000')
    args = parser.parse_args()

    from dataset import ART

    art = ART()
    art.load_data()
    print(len(art.filenames))

    detection_model, recognition_model, label_dict = init_ocr_model()
    log_path = '/data/models/text_recognition/AttentionOCR/log_art_baseline.txt'
    f = open(log_path, 'w')
    eval_crop(detection_model, recognition_model, args, art.filenames, art.points, art.transcripts, label_dict, f)
    f.close()


def test_2019icdar_art_test_task3():
    # parser = argparse.ArgumentParser(description='OCR')
    # parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model',
    #                     default='./checkpoint/model-10000')
    # args = parser.parse_args()
    #
    # from dataset import ART
    #
    # art = ART()
    # art.load_data()
    # print(len(art.filenames))
    log_path = '/data/datasets/text_recognition/ICDAR2019/art/task3/end_2_end_result.json'
    img_paths = glob.glob('/data/datasets/text_recognition/ICDAR2019/art/test_part1_images/*') + \
                glob.glob('/data/datasets/text_recognition/ICDAR2019/art/test_part2_images/*')

    detection_model, recognition_model, label_dict = init_ocr_model()
    result = dict()
    for i, img_path in enumerate(img_paths):
        t_start = time.time()
        result_dict = test_intigrate(detection_model, recognition_model, img_path, label_dict)
        t_end = time.time()
        result.update(result_dict)
        print("image {}: {}, time costs {}s.".format(i + 1, img_path, t_end - t_start))
    json.dump(result, open(log_path, 'w'))
    # f = open(log_path, 'w')
    # f.close()


if __name__ == '__main__':
    # test_2019icdar_art_train_task2()
    test_2019icdar_art_test_task3()