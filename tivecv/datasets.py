# Modified by Zilong Jia from https://github.com/dbolya/tide
import sys

import numpy as np

from .data import TiveData

import zipfile
from pathlib import Path
from appdirs import user_data_dir
import urllib.request
from collections import defaultdict
import shutil
import json
import os

from tidecv.datasets import default_name
from tidecv import functions as f

# Suppoort video datasets with the same annotation format like YTVIS and OVIS

def VideoData(path: str = None, name: str = None) -> TiveData:
    """
    Loads ground truth from a ytvis-style annotation file.
    """
    if name is None: name = default_name(path)

    with open(path, 'r') as json_file:
        ytvisjson = json.load(json_file)

    videos = ytvisjson['videos']
    anns = ytvisjson['annotations']
    cats = ytvisjson['categories'] if 'categories' in ytvisjson else None

    # Add everything from the ytvis json into our data structure
    data = TiveData(name, max_dets=100)

    video_lookup = {}

    for idx, video in enumerate(videos):
        video_lookup[video['id']] = video
        data.add_image(video['id'], video['file_names'])

    if cats is not None:
        for cat in cats:
            data.add_class(cat['id'], cat['name'])

    for ann in anns:
        video = ann['video_id']
        _class = ann['category_id']
        box = ann['bboxes']
        gt_len = 0
        mask = []
        for seg in ann['segmentations']:
            mask.append(
                f.toRLE(seg, video_lookup[video]['width'], video_lookup[video]['height']) if seg != None else None)

        for _m in mask:
            if _m != None:
                gt_len += 1

        if ann['iscrowd']:
            data.add_ignore_region(video, _class, box, mask, gt_length=gt_len)
        else:
            data.add_ground_truth(video, _class, box, mask, gt_length=gt_len)
    return data


def VideoDataResult(path: str, name: str = None) -> TiveData:
    """ Loads predictions from a ytvis-style results file. """
    if name is None: name = default_name(path)

    with open(path, 'r') as json_file:
        dets = json.load(json_file)

    data = TiveData(name)

    for det in dets:
        video = det['video_id']
        _cls = det['category_id']
        score = det['score']
        box = det['bbox'] if 'bbox' in det else None
        mask = det['segmentations'] if 'segmentations' in det else None
        mask_len = 0
        if mask != None:
            for _m in mask:
                if np.any(_m):
                    mask_len += 1

        data.add_detection(video, _cls, score, box, mask, gt_length=mask_len)

    return data


def VideoData_perimg(path: str = None, name: str = None) -> TiveData:
    """
    Loads ground truth from a ytvis-style annotation file.
    """
    if name is None: name = default_name(path)

    with open(path, 'r') as json_file:
        ytvisjson = json.load(json_file)

    videos = ytvisjson['videos']
    anns = ytvisjson['annotations']
    cats = ytvisjson['categories'] if 'categories' in ytvisjson else None

    # Add everything from the ytvis json into our data structure
    data = TiveData(name, max_dets=100)

    video_lookup = {}
    im_id = 0

    for idx, video in enumerate(videos):
        video_lookup[video['id']] = video
        video_lookup[video['id']]['im_id'] = []
        for im_f in video['file_names']:
            data.add_image(im_id, im_f)
            video_lookup[video['id']]['im_id'].append(im_id)
            im_id += 1
    data.video_lookup = video_lookup

    if cats is not None:
        for cat in cats:
            data.add_class(cat['id'], cat['name'])

    for ann in anns:
        video = ann['video_id']
        _class = ann['category_id']
        box = ann['bboxes']
        mask = []
        for seg in ann['segmentations']:
            mask.append(
                f.toRLE(seg, video_lookup[video]['width'], video_lookup[video]['height']) if seg != None else None)
        for i_m, (m, b) in enumerate(zip(mask, box)):
            image = video_lookup[video]['im_id'][i_m]
            if m != None:
                if ann['iscrowd']:
                    data.add_ignore_region(image, _class, b, m)
                else:
                    data.add_ground_truth(image, _class, b, m)
    return data


def VideoDataResult_perimg(path: str, data_ann: TiveData, name: str = None) -> TiveData:
    """ Loads predictions from a ytvis-style results file. """
    if name is None: name = default_name(path)

    with open(path, 'r') as json_file:
        dets = json.load(json_file)

    data = TiveData(name)

    for det in dets:
        video = det['video_id']
        _cls = det['category_id']
        score = det['score']
        box = det['bbox'] if 'bbox' in det else [None] * len(det['segmentations'])
        mask = det['segmentations'] if 'segmentations' in det else None

        for i_m, (b, m) in enumerate(zip(box, mask)):
            image = data_ann.video_lookup[video]['im_id'][i_m]
            data.add_detection(image, _cls, score, b, m)

    return data
