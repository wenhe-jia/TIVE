# Modified by Zilong Jia from https://github.com/dbolya/tide
import os

from collections import defaultdict
import numpy as np
import cv2

from tidecv import Data

from tidecv import functions as f


class TiveData(Data):
    """
    A class to hold ground truth or predictions data in an easy to work with format.
    Note that any time they appear, bounding boxes are [x, y, width, height] and masks
    are either a list of polygons or pycocotools RLEs.

    Also, don't mix ground truth with predictions. Keep them in separate data objects.

    'max_dets' specifies the maximum number of detections the model is allowed to output for a given image.
    """

    def __init__(self, name: str, max_dets: int = 100):
        super().__init__(name, max_dets)

        self.video_lookup = None

    def _add(self, image_id: int, class_id: int, box: object = None, mask: object = None, score: float = 1,
             ignore: bool = False, gt_length: int = None):
        """ Add a data object to this collection. You should use one of the below functions instead. """
        self._make_default_class(class_id)
        self._make_default_image(image_id)
        new_id = len(self.annotations)

        self.annotations.append({
            '_id': new_id,
            'score': score,
            'image': image_id,
            'class': class_id,
            'bbox': self._prepare_box(box),
            'mask': self._prepare_mask(mask),
            'ignore': ignore,
            'gt_length': gt_length,
        })

        self.images[image_id]['anns'].append(new_id)

    def add_ground_truth(self, image_id: int, class_id: int, box: object = None, mask: object = None,
                         gt_length: int = None):
        """ Add a ground truth. If box or mask is None, this GT will be ignored for that mode. """
        self._add(image_id, class_id, box, mask, gt_length=gt_length)

    def add_detection(self, image_id: int, class_id: int, score: int, box: object = None, mask: object = None,
                      gt_length: int = None, ignore: bool = False):
        """ Add a predicted detection. If box or mask is None, this prediction will be ignored for that mode. """
        self._add(image_id, class_id, box, mask, score=score, gt_length=gt_length, ignore=ignore)

    def add_ignore_region(self, image_id: int, class_id: int = None, box: object = None, mask: object = None,
                          gt_length: int = None):
        """
        Add a region inside of which background detections should be ignored.
        You can use these to mark a region that has deliberately been left unannotated
        (e.g., if is a huge crowd of people and you don't want to annotate every single person in the crowd).

        If class_id is -1, this region will match any class. If the box / mask is None, the region will be the entire image.
        """
        self._add(image_id, class_id, box, mask, ignore=True, gt_length=gt_length)

