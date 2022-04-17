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

YTVIS_CATEGORIES_2021 = {
    1: "airplane",
    2: "bear",
    3: "bird",
    4: "boat",
    5: "car",
    6: "cat",
    7: "cow",
    8: "deer",
    9: "dog",
    10: "duck",
    11: "earless_seal",
    12: "elephant",
    13: "fish",
    14: "flying_disc",
    15: "fox",
    16: "frog",
    17: "giant_panda",
    18: "giraffe",
    19: "horse",
    20: "leopard",
    21: "lizard",
    22: "monkey",
    23: "motorbike",
    24: "mouse",
    25: "parrot",
    26: "person",
    27: "rabbit",
    28: "shark",
    29: "skateboard",
    30: "snake",
    31: "snowboard",
    32: "squirrel",
    33: "surfboard",
    34: "tennis_racket",
    35: "tiger",
    36: "train",
    37: "truck",
    38: "turtle",
    39: "whale",
    40: "zebra",
}


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
        default="configs/youtubevis_2021mini/video_maskformer2_R50_bs16_8ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--result-file',
        default='/home/jwh/vis/mini360relate/results_minioriginal.json',
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


def toRLE(mask: object, w: int, h: int):
    """
    Borrowed from Pycocotools:
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """

    if type(mask) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(mask, h, w)
        return mask_util.merge(rles)
    elif type(mask['counts']) == list:
        # uncompressed RLE
        return mask_util.frPyObjects(mask, h, w)
    else:
        return mask


def iou_seq(d_seq, g_seq):
    '''

    :param d_seq: RLE object
    :param g_seq: RLE object
    :return:
    '''
    i = .0
    u = .0
    for d, g in zip(d_seq, g_seq):
        if d and g:
            i += mask_util.area(mask_util.merge([d, g], True))
            u += mask_util.area(mask_util.merge([d, g], False))
        elif not d and g:
            u += mask_util.area(g)
        elif d and not g:
            u += mask_util.area(d)
    # if not u > .0:
    #     print("Mask sizes in video  and category  may not match!")
    iou = i / u if u > .0 else .0
    return iou


def get_predictions_from_json(json_file):
    pre_list = {}
    # fr = {'video_id', 'score', 'category_id', 'segmentations'}
    for fr in tqdm(json_file):

        if fr['video_id'] not in pre_list:
            pre_list[fr['video_id']] = []

        # instance = {"image_size": None, "pred_scores": None, "pred_labels": None,
        #             "pred_masks": None, "instance_id": None}

        vid = fr['video_id']

        fr["instance_id"] = len(pre_list[vid]) + 1

        # instance["pred_scores"] = fr["score"]
        # fr['category_id'] = fr['category_id']
        # mask = [mask_util.decode(_m) for _m in fr['segmentations']]
        # mask = fr['segmentations']
        # instance["pred_masks"] = mask
        # fr['segmentations'] = [mask_util.decode(_m) for _m in fr['segmentations']]
        fr["image_size"] = fr['segmentations'][0]['size']
        pre_list[vid].append(fr)
    return pre_list


def get_groundtruth_from_json(json_file):
    pre_list = {}
    # fr = {'video_id', 'iscrowd', 'height', 'width', 'length', 'segmentations', 'bboxes', 'category_id', 'id', 'areas'}
    for fr in tqdm(json_file):
        if fr['video_id'] not in pre_list:
            pre_list[fr['video_id']] = []
        fr["instance_id"] = len(pre_list[fr['video_id']]) + 1

        # print(fr['segmentations'])
        mask = []
        for seg in fr['segmentations']:
            mask.append(
                toRLE(seg, fr['width'], fr[
                    'height']) if seg != None else
                mask_util.encode(np.array(np.zeros((fr['height'], fr['width']))[:, :, None], order="F", dtype="uint8"))[
                    0])
        fr['rle_segmentations'] = mask

        # fr['segmentations'] = []
        # for _m in fr['rle_segmentations']:
        #     if _m == None:
        #         _m = np.zeros((fr['height'], fr['width']))
        #     fr['segmentations'].append(mask_util.decode(_m))
        ''''''
        fr['segmentations'] = [mask_util.decode(_m) for _m in fr['rle_segmentations']]

        pre_list[fr['video_id']].append(fr)

    return pre_list


def draw_instance_id(im_in, insid):
    im_in[:22, :60, :] = 0  # np.zeros((10,10,3))
    im = cv2.putText(np.ascontiguousarray(im_in), 'id' + str(insid), (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     (255, 255, 255), 1)
    return im


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
    '''
    Metadata(
    evaluator_type='ytvis', 
    image_root='datasets/ytvis_2021_mini/valid/JPEGImages', 
    json_file='datasets/ytvis_2021_mini/valid.json', 
    name='ytvis_2021_mini_val', thing_classes=['airplane', ...], 
    thing_colors=[[106, 0, 228], ...], 
    thing_dataset_id_to_contiguous_id={1: 0,...})
    '''

    video_json = metadata.get('json_file')
    video_json = json.load(open(video_json, 'r'))
    groundtruth = get_groundtruth_from_json(video_json['annotations'])

    for vid, v in enumerate(video_json['videos']):
        print('processing video', vid)

        '''preprocess'''

        img_path_root = metadata.get('image_root')
        pred = predictions[v['id']]
        anno = groundtruth[v['id']]
        for tmp_g in anno:
            tmp_g['ignore'] = tmp_g['ignore'] if 'ignore' in tmp_g else 0
            tmp_g['ignore'] = 'iscrowd' in tmp_g and tmp_g['iscrowd']
        for tmp_g in anno:
            if tmp_g['ignore']:
                tmp_g['_ignore'] = 1
            else:
                tmp_g['_ignore'] = 0

        '''calculate iou'''

        inds = np.argsort([-d['score'] for d in pred], kind='mergesort')
        dt = [pred[i] for i in inds]
        g = [g['rle_segmentations'] for g in anno]
        gtind = np.argsort([gind['_ignore'] for gind in anno], kind='mergesort')
        g = [g[i] for i in gtind]
        gt = [anno[i] for i in gtind]
        d = [d['segmentations'] for d in dt]
        iscrowd = [int(o['iscrowd']) for o in gt]

        ious = np.zeros([len(pred), len(anno)])
        for i, j in np.ndindex(ious.shape):
            ious[i, j] = iou_seq(d[i], g[j])

        '''load frames'''

        vid_frames = []
        for path in v['file_names']:
            path = os.path.join(img_path_root, path)
            img = read_image(path, format="BGR")
            vid_frames.append(img)

        path_root = os.path.join(args.output, 'video' + str(vid))

        '''draw gt'''

        print('drawing gt')
        path_gt_root = os.path.join(path_root, 'gt')
        image_size = pred[0]["image_size"]
        gt_labels = [_a["category_id"] - 1 for _a in anno]
        gt_masks = [_a["segmentations"] for _a in anno]
        gt_ins_id = [_a["instance_id"] for _a in anno]
        for gl, gm, giid in zip(gt_labels, gt_masks, gt_ins_id):
            gt_frame_masks = gm
            # print(gt_frame_masks)
            # cat_id+_+category
            path_gt_root_i = os.path.join(path_gt_root, str(gl + 1) + '_' + YTVIS_CATEGORIES_2021[gl + 1])
            os.makedirs(path_gt_root_i, exist_ok=True)
            for frame_idx in range(len(vid_frames)):
                frame = vid_frames[frame_idx][:, :, ::-1]
                visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                ins = Instances(image_size)
                ins.scores = [1]
                ins.pred_classes = [gl]
                gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                vis_output = visualizer.draw_instance_predictions(predictions=ins)
                vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1], giid)
                cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.png'),
                            vis_im)
            print('successfully saved gt')

        # calucate iou
        # iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

        '''match'''

        iouThrs = [0.1, 0.5]

        T = len(iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))

        dtm_01 = np.zeros((T, D))  # iou<0.1 fp
        gtm_index_01 = np.zeros((T, D))
        gtm_01 = np.zeros((T, G))
        dtm_015 = np.zeros((T, D))  # 0.1<iou<0.5 fp
        gtm_index_015 = np.zeros((T, D))
        gtm_015 = np.zeros((T, G))
        dtm_05 = np.zeros((T, D))  # iou>0.5 fp
        gtm_05 = np.zeros((T, G))
        gtm_index_05 = np.zeros((T, D))

        gtm_index = np.zeros((T, D))  # store the index of gt that corespond to dtm

        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))

        for tind, t in enumerate(iouThrs):
            for dind, d1 in enumerate(dt):

                iou = min([t, 1 - 1e-10])
                m = -1
                m01 = -1
                m015 = -1
                m05 = -1
                for gind, g1 in enumerate(gt):

                    # if ious[dind, gind] < iou:
                    #     if ious[dind, gind] < 0.1:
                    #         if m01 == -1:
                    #             iou01 = ious[dind, gind]
                    #             m01 = gind
                    #         if ious[dind, gind] > iou01:
                    #             iou01 = ious[dind, gind]
                    #             m01 = gind
                    #     else:
                    #         if m015 == -1:
                    #             iou015 = ious[dind, gind]
                    #             m015 = gind
                    #         if ious[dind, gind] > iou015:
                    #             iou015 = ious[dind, gind]
                    #             m015 = gind

                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue

                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break

                    if ious[dind, gind] < iou:
                        continue

                    iou = ious[dind, gind]
                    m = gind

                # if m01 != -1:
                #     dtm_01[tind, dind] = gt[m01]['instance_id']
                #     gtm_index_01[tind, dind] = m01
                #     gtm_01[tind, m01] = d1['instance_id']
                #
                # if m015 != -1:
                #     dtm_015[tind, dind] = gt[m015]['instance_id']
                #     gtm_index_015[tind, dind] = m015
                #     gtm_015[tind, m015] = d1['instance_id']

                if m == -1:
                    for gind, g1 in enumerate(gt):

                        if ious[dind, gind] < iou:
                            if ious[dind, gind] < 0.1:
                                if m01 == -1:
                                    iou01 = ious[dind, gind]
                                    m01 = gind
                                if ious[dind, gind] > iou01:
                                    iou01 = ious[dind, gind]
                                    m01 = gind
                            else:
                                if m015 == -1:
                                    iou015 = ious[dind, gind]
                                    m015 = gind
                                if ious[dind, gind] > iou015:
                                    iou015 = ious[dind, gind]
                                    m015 = gind
                        else:
                            if m05 == -1:
                                iou05 = ious[dind, gind]
                                m05 = gind
                            if ious[dind, gind] > iou05:
                                iou05 = ious[dind, gind]
                                m05 = gind
                    if m01 != -1:
                        dtm_01[tind, dind] = gt[m01]['instance_id']
                        gtm_index_01[tind, dind] = m01
                        gtm_01[tind, m01] = d1['instance_id']

                    if m015 != -1:
                        dtm_015[tind, dind] = gt[m015]['instance_id']
                        gtm_index_015[tind, dind] = m015
                        gtm_015[tind, m015] = d1['instance_id']

                    if m05 != -1:
                        dtm_05[tind, dind] = gt[m05]['instance_id']
                        gtm_index_05[tind, dind] = m05
                        gtm_05[tind, m015] = d1['instance_id']
                    continue

                if d1['category_id'] != gt[m]['category_id']:
                    dtm_05[tind, dind] = gt[m]['instance_id']
                    gtm_index_05[tind, dind] = m
                    gtm_05[tind, m] = d1['instance_id']
                    continue

                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['instance_id']
                gtm[tind, m] = d1['instance_id']
                gtm_index[tind, dind] = m

        dtIg = np.logical_or(dtIg, dtm == 0)

        '''draw tp'''

        path_tp_root = os.path.join(path_root, 'tp')
        # tp_match [(dt_index,gt_id)]
        tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm[1]) if _dtm != 0]
        # print(dtm[1])
        # print(tp_match)
        # sys.exit()
        tp_labels = []
        tp_masks = []
        tp_ins_id = []
        tp_matchgt_insid = []  # the gt instance_id that matched to the dt
        tp_matchgt_index = []  # the gt index that matched to the dt
        tp_dt_index = []
        tp_dt_score = []
        for (tpm_i, tpm_dtm) in tp_match:
            tp_labels.append(dt[tpm_i]['category_id'] - 1)
            tp_masks.append(dt[tpm_i]['segmentations'])
            tp_ins_id.append(dt[tpm_i]['instance_id'])
            tp_matchgt_insid.append(tpm_dtm)
            tp_matchgt_index.append(gtm_index[1, tpm_i])
            tp_dt_index.append(tpm_i)
            tp_dt_score.append(dt[tpm_i]['score'])

        image_size = pred[0]["image_size"]
        # tp_labels = [_a["category_id"] - 1 for _a in anno]
        # tp_masks = [_a["segmentations"] for _a in anno]
        # tp_ins_id = [_a["instance_id"] for _a in anno]
        print('--num_tp', len(tp_labels))
        for gl, gm, giid, matchid, tpidx, matchidx, tpscore in zip(tp_labels, tp_masks, tp_ins_id, tp_matchgt_insid,
                                                                   tp_dt_index, tp_matchgt_index, tp_dt_score):
            print('drawing tp')
            gt_frame_masks = [mask_util.decode(_m) for _m in gm]
            # print(gt_frame_masks)
            # iou + _ + cat_id + _ + category + score
            path_gt_root_i = os.path.join(path_tp_root,
                                          str(ious[tpidx, int(matchidx)]) + '_' + str(gl + 1) + '_' +
                                          YTVIS_CATEGORIES_2021[
                                              gl + 1] + str(tpscore))
            os.makedirs(path_gt_root_i, exist_ok=True)
            for frame_idx in range(len(vid_frames)):
                frame = vid_frames[frame_idx][:, :, ::-1]
                visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                ins = Instances(image_size)
                ins.scores = [tpscore]
                ins.pred_classes = [gl]
                gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                vis_output = visualizer.draw_instance_predictions(predictions=ins)
                vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1], str(giid) + '-' + str(matchid))
                cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.png'),
                            vis_im)
            print('successfully saved tp')

        '''draw fp  iou<0.1 | 0.1< iou<0.5 | iou>0.5  '''

        for k in range(3):

            # tp_match [(dt_index,gt_id)]
            # tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm[1]) if _dtm != 0]
            tp_labels = []
            tp_masks = []
            tp_ins_id = []
            tp_matchgt_insid = []  # the gt instance_id that matched to the dt
            tp_matchgt_index = []  # the gt index that matched to the dt
            tp_dt_index = []
            tp_dt_score = []

            if k == 0:
                path_tp_root = os.path.join(path_root, 'fp', 'iou<0.1')
                tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm_01[1]) if _dtm != 0]
            elif k == 1:
                path_tp_root = os.path.join(path_root, 'fp', '0.1<iou<0.5')
                tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm_015[1]) if _dtm != 0]
            else:
                path_tp_root = os.path.join(path_root, 'fp', 'iou>0.5')
                tp_match = [(_i, _dtm) for _i, _dtm in enumerate(dtm_05[1]) if _dtm != 0]

            for (tpm_i, tpm_dtm) in tp_match:
                tp_labels.append(dt[tpm_i]['category_id'] - 1)
                tp_masks.append(dt[tpm_i]['segmentations'])
                tp_ins_id.append(dt[tpm_i]['instance_id'])
                tp_matchgt_insid.append(tpm_dtm)
                if k == 0:
                    tp_matchgt_index.append(gtm_index_01[1, tpm_i])
                elif k == 1:
                    tp_matchgt_index.append(gtm_index_015[1, tpm_i])
                else:
                    tp_matchgt_index.append(gtm_index_05[1, tpm_i])

                tp_dt_index.append(tpm_i)
                tp_dt_score.append(dt[tpm_i]['score'])

            image_size = pred[0]["image_size"]
            # tp_labels = [_a["category_id"] - 1 for _a in anno]
            # tp_masks = [_a["segmentations"] for _a in anno]
            # tp_ins_id = [_a["instance_id"] for _a in anno]
            print('--num_fp ' + str(k), len(tp_labels))
            for gl, gm, giid, matchid, tpidx, matchidx, tpscore in zip(tp_labels, tp_masks, tp_ins_id, tp_matchgt_insid,
                                                                       tp_dt_index, tp_matchgt_index, tp_dt_score):
                print('drawing fp ' + str(k))
                gt_frame_masks = [mask_util.decode(_m) for _m in gm]
                # print(gt_frame_masks)
                # iou + _ + cat_id + _ + category + score
                path_gt_root_i = os.path.join(path_tp_root,
                                              str(ious[tpidx, int(matchidx)]) + '_' + str(gl + 1) + '_' +
                                              YTVIS_CATEGORIES_2021[
                                                  gl + 1] + str(tpscore))
                os.makedirs(path_gt_root_i, exist_ok=True)
                for frame_idx in range(len(vid_frames)):
                    frame = vid_frames[frame_idx][:, :, ::-1]
                    visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
                    ins = Instances(image_size)
                    ins.scores = [tpscore]
                    ins.pred_classes = [gl]
                    gt_frame_masks[frame_idx] = [torch.from_numpy(gt_frame_masks[frame_idx])]
                    ins.pred_masks = torch.stack(gt_frame_masks[frame_idx], dim=0)

                    vis_output = visualizer.draw_instance_predictions(predictions=ins)
                    vis_im = draw_instance_id(vis_output.get_image()[:, :, ::-1], str(giid) + '-' + str(matchid))
                    cv2.imwrite(os.path.join(path_gt_root_i, 'frame' + str(frame_idx) + '.png'),
                                vis_im)
                print('successfully saved fp ' + str(k))
