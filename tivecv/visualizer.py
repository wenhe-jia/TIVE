# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File ：visualizer.py
@Author ：jzl
@Date ：2022/4/30 17:44 
'''
import copy
import os, sys
import cv2
import numpy as np
import pycocotools.mask as mask_utils

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
YTVIS_COLOR_2021 = {
    1: [106, 0, 228],
    2: [174, 57, 255],
    3: [255, 109, 65],
    4: [0, 0, 192],
    5: [0, 0, 142],
    6: [255, 77, 255],
    7: [120, 166, 157],
    8: [209, 0, 151],
    9: [0, 226, 252],
    10: [179, 0, 194],
    11: [174, 255, 243],
    12: [110, 76, 0],
    13: [73, 77, 174],
    14: [250, 170, 30],
    15: [0, 125, 92],
    16: [107, 142, 35],
    17: [0, 82, 0],
    18: [72, 0, 118],
    19: [182, 182, 255],
    20: [255, 179, 240],
    21: [119, 11, 32],
    22: [0, 60, 100],
    23: [0, 0, 230],
    24: [130, 114, 135],
    25: [165, 42, 42],
    26: [220, 20, 60],
    27: [100, 170, 30],
    28: [183, 130, 88],
    29: [134, 134, 103],
    30: [5, 121, 0],
    31: [133, 129, 255],
    32: [188, 208, 182],
    33: [145, 148, 174],
    34: [255, 208, 186],
    35: [166, 196, 102],
    36: [0, 80, 100],
    37: [0, 0, 70],
    38: [0, 143, 149],
    39: [0, 228, 0],
    40: [199, 100, 0],
}


def draw_information(im_in1, insid, is_gt=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    margin = 5
    thickness = 1

    im_in = copy.deepcopy(im_in1)
    # BGR for three bad cases
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    if is_gt:
        color = (255, 105, 65)
    else:
        color = (13, 23, 227)

    x = y = 0

    for _nt, text in enumerate(insid):
        size = cv2.getTextSize(text, font, font_scale, thickness)

        text_width = size[0][0]
        text_height = size[0][1]

        im_in[y:text_height + margin + y, :text_width + margin, :] = np.array(
            [220, 220, 220])  # np.zeros((10,10,3))

        x = margin
        if _nt == 0:
            y = text_height + y
        else:
            y = text_height + y + margin

        im_in = cv2.putText(np.ascontiguousarray(im_in), text, (x, y), font, font_scale, color, thickness)
    return im_in


def toRLE(mask: object, w: int, h: int):
    """
    Borrowed from Pycocotools:
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """

    if type(mask) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(mask, h, w)
        return mask_utils.merge(rles)
    elif type(mask['counts']) == list:
        # uncompressed RLE
        return mask_utils.frPyObjects(mask, h, w)
    else:
        return mask


# visualizer for vis
class Visualizer:
    def __init__(self, ex, video_id, video_files, image_root, save_root):
        self.save_root = save_root
        self.ex = ex
        self.video_id = video_id
        self.video_files = video_files
        self.image_root = image_root
        if self.image_root != None:
            self._read_images()

    def _read_images(self):
        self.images = []
        for fname in self.video_files:
            self.images.append(cv2.imread(os.path.join(self.image_root, fname)))

        self.width = self.images[0].shape[1]
        self.height = self.images[0].shape[0]

    def _coloring_mask(self, mask, img1, color, alpha):
        '''

        Args:
            mask:
            img:
        Returns: video with masks

        '''
        img = copy.deepcopy(img1)
        color = np.array(color, dtype=np.uint8)
        mask = mask.astype(np.bool)
        img[mask] = img[mask] * (1 - alpha) + alpha * color
        return img

    # draw instance prediction by errortype
    def draw(self, pred, error_type):
        if self.image_root != None:

            if error_type != 'Miss':
                print('--processing video:', self.video_id, '  prediction:', pred['_id'], '  save type:', error_type)
                # creat folder
                save_path = os.path.join(self.save_root, 'video' + str(self.video_id), error_type, )
            else:
                print('--processing video:', self.video_id, '  gt:', pred, '  save type:', error_type)
                # creat folder
                save_path = os.path.join(self.save_root, 'video' + str(self.video_id), error_type, )

            os.makedirs(save_path, exist_ok=True)

            _files = list(os.listdir(save_path))
            if error_type != 'Miss':
                save_path = os.path.join(save_path,
                                         str(len(_files)) + '_score-' + str(round(pred['score'], 2)) + '_iou-' +
                                         str(round(pred['iou'], 2)))
                # generate color
                colors = [x for x in YTVIS_COLOR_2021[pred['class']]][::-1]
            else:
                save_path = os.path.join(save_path, str(len(_files)) + '_gtid-' + str(self.ex.gt[pred]['_id']))
                # generate color
                colors = [x for x in YTVIS_COLOR_2021[self.ex.gt[pred]['class']]][::-1]
            os.makedirs(save_path, exist_ok=True)

            alpha = 0.5

            # print('get masks')

            masks = [None, None]  # [gt_mask,dt_mask]
            if error_type == 'Miss':
                masks[0] = self.ex.gt[pred]['mask']
                txt_gt = ['gt_id:' + str(self.ex.gt[pred]['_id']),
                          'label:' + YTVIS_CATEGORIES_2021[self.ex.gt[pred]['class']],
                          'used:' + str(self.ex.gt[pred]['used'])]

            elif len(self.ex.gt) != 0 and pred['vis_gt_idx'] != None:
                masks[0] = self.ex.gt[pred['vis_gt_idx']]['mask']
                txt_gt = ['gt_id:' + str(self.ex.gt[pred['vis_gt_idx']]['_id']),
                          'label:' + YTVIS_CATEGORIES_2021[self.ex.gt[pred['vis_gt_idx']]['class']],
                          'used:' + str(self.ex.gt[pred['vis_gt_idx']]['used'])]
            # if pred exists
            if error_type != 'Miss':
                masks[1] = pred['mask']

                txt_pred = ['label:' + YTVIS_CATEGORIES_2021[pred['class']], 'iou:' + str(round(pred['iou'], 2)),
                            'score:' + str(round(pred['score'], 2))]

            # print('drawing')
            # get masks and dets
            if masks[0] != None:
                gtmask = []
                for idx, gtm in enumerate(masks[0]):
                    if gtm != None:
                        gtm = mask_utils.decode(toRLE(gtm, self.width, self.height))
                    else:
                        gtm = np.zeros((self.height, self.width))
                    gtm = self._coloring_mask(gtm, self.images[idx], colors, alpha)
                    gtmask.append(draw_information(gtm, txt_gt, True))

                masks[0] = gtmask
            # if pred exists
            if error_type != 'Miss':
                dtmask = []
                for idx, dtm in enumerate(masks[1]):
                    if dtm == None:
                        dtm = np.zeros((self.height, self.width))
                    else:
                        dtm = mask_utils.decode(dtm)
                    dtm = self._coloring_mask(dtm, self.images[idx], colors, alpha)
                    dtmask.append(draw_information(dtm, txt_pred))
                masks[1] = dtmask

            # print('saving')
            # save images
            final_imgs = []

            # gt and perd both exist
            if masks[0] != None and error_type != 'Miss':
                for dtm, gtm in zip(masks[1], masks[0]):
                    final_imgs.append(np.concatenate([dtm, (np.ones_like(dtm) * 255)[:10], gtm], axis=0))
            # only pred exists
            elif masks[0] == None and error_type != 'Miss':
                final_imgs = masks[1]
            # only gt exists
            elif masks[0] != None and error_type == 'Miss':
                final_imgs = masks[0]

            for fnum, fim in enumerate(final_imgs):
                cv2.imwrite(os.path.join(save_path, 'frame-' + str(fnum) + '.jpg'), fim)
