# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File ：visualize.py
@Author ：jzl
@Date ：2022/3/12 11:41 
'''

import argparse
import json
import os
import sys
import cv2
import torch
from tqdm import tqdm
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from torch.cuda.amp import autocast

import pycocotools.mask as mask_util

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from visualizer import TrackVisualizer

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 visualizer")
    parser.add_argument(
        "--config-file",
        default="configs/youtubevis_2021/video_maskformer2_R50_bs16_8ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--result-file',
        default='output/inference/results.json',
        help='path to result json file',
    )

    parser.add_argument(
        "--output",
        default='visualize_output',
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--save-frames",
        default=False,
        help="Save frame level image outputs.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    return parser


def get_predictions_from_json(json_file):
    pre_list = {}
    # predictions = {"image_size": (720, 1280), "pred_scores": [], "pred_labels": [], "pred_masks": []}
    for fr in tqdm(json_file):
        if fr['video_id'] not in pre_list:
            pre_list[fr['video_id']] = {"image_size": (720, 1280), "pred_scores": [], "pred_labels": [],
                                        "pred_masks": []}
        vid = fr['video_id']

        pre_list[vid]["pred_scores"].append(fr["score"])
        pre_list[vid]["pred_labels"].append(fr['category_id']-1)
        mask = [mask_util.decode(_m) for _m in fr['segmentations']]
        pre_list[vid]["pred_masks"].append(mask)


    return pre_list


if __name__ == '__main__':

    args = get_parser().parse_args()

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    cfg = setup_cfg(args)

    # load jsons and get predictions
    res_json = json.load(open(args.result_file, 'r'))
    print('loading predictions')
    predictions = get_predictions_from_json(res_json)


    # visualize
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    # print(metadata)
    video_json = metadata.get('json_file')
    video_json = json.load(open(video_json, 'r'))

    for vid, v in tqdm(enumerate(video_json['videos'])):
        img_path_root = metadata.get('image_root')
        pred = predictions[v['id']]
        # load frames
        vid_frames = []
        for path in v['file_names']:
            path = os.path.join(img_path_root, path)
            img = read_image(path, format="BGR")
            vid_frames.append(img)

        # draw
        image_size = pred["image_size"]
        pred_scores = pred["pred_scores"]
        pred_labels = pred["pred_labels"]
        pred_masks = pred["pred_masks"]

        frame_masks = list(zip(*pred_masks))
        visualized_output = []
        for frame_idx in range(len(vid_frames)):
            frame = vid_frames[frame_idx][:, :, ::-1]
            visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
            ins = Instances(image_size)
            if len(pred_scores) > 0:
                ins.scores = pred_scores
                ins.pred_classes = pred_labels
                frame_masks[frame_idx] = [torch.from_numpy(pm) for pm in frame_masks[frame_idx]]
                ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)

            vis_output = visualizer.draw_instance_predictions(predictions=ins)
            visualized_output.append(vis_output)

        H, W = visualized_output[0].height, visualized_output[0].width
        cap = cv2.VideoCapture(-1)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(args.output, "visualization" + str(vid) + ".mp4"), fourcc, 10.0, (W, H),
                              True)
        for _vis_output in visualized_output:
            frame = _vis_output.get_image()[:, :, ::-1]
            out.write(frame)
        cap.release()
        out.release()
