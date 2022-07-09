# Modified by Zilong Jia from https://github.com/dbolya/tide
import sys

from .data import TiveData
from .errors.main_errors import *
from .visualizer import Visualizer
from . import plotting as P

from pycocotools import mask as mask_utils
from collections import defaultdict, OrderedDict
import numpy as np
from typing import Union
import os, math
from tqdm import tqdm

from tidecv.ap import ClassedAPDataObject
from tidecv.errors.main_errors import *
from tidecv.errors.qualifiers import Qualifier, AREA
from tidecv.quantify import TIDE, TIDEExample, TIDERun
from tidecv import functions as f


class TIVEExample(TIDEExample):
    """ Computes all the data needed to evaluate a set of predictions and gt for a single image. """

    def __init__(self, preds: list, gt: list, pos_thresh: float, mode: str, max_dets: int, run_errors: bool = True,
                 isvideo: bool = False):
        self.pred_ignore = [x for x in preds if x['ignore']]
        self.isvideo = isvideo
        super().__init__(preds, gt, pos_thresh, mode, max_dets, run_errors)

    def _run(self):
        preds = self.preds
        gt = self.gt
        ignore = self.ignore_regions
        det_type = 'bbox' if self.mode == TIDE.BOX else 'mask'
        max_dets = self.max_dets

        if len(preds) == 0:
            raise RuntimeError('Example has no predictions!')

        # Sort descending by score
        preds.sort(key=lambda pred: -pred['score'])
        preds = preds[:max_dets]
        self.preds = preds  # Update internally so TIDERun can update itself if :max_dets takes effect
        detections = [x[det_type] for x in preds]

        def iou_seq(d_seq, g_seq):
            i = .0
            u = .0
            for d, g in zip(d_seq, g_seq):
                if d and g:
                    i += mask_utils.area(mask_utils.merge([d, g], True))
                    u += mask_utils.area(mask_utils.merge([d, g], False))
                elif not d and g:
                    u += mask_utils.area(g)
                elif d and not g:
                    u += mask_utils.area(d)
            iou = i / u if u > .0 else .0
            return iou

        # IoU is [len(detections), len(gt)]

        if self.isvideo:
            self.gt_iou = np.zeros((len(detections), len(gt)))
            for dind, di in enumerate(detections):
                for gind, gi in enumerate([x[det_type] for x in gt]):
                    self.gt_iou[dind, gind] = iou_seq(di, gi)

        else:

            self.gt_iou = mask_utils.iou(
                detections,
                [x[det_type] for x in gt],
                [False] * len(gt))

        # Store whether a prediction / gt got used in their data list
        # Note: this is set to None if ignored, keep that in mind
        for idx, pred in enumerate(preds):
            pred['used'] = False
            pred['_idx'] = idx
            pred['iou'] = 0
        for idx, truth in enumerate(gt):
            truth['used'] = False
            truth['usable'] = False
            truth['_idx'] = idx

        pred_cls = np.array([x['class'] for x in preds])
        gt_cls = np.array([x['class'] for x in gt])

        if len(gt) > 0:
            # A[i,j] is true if the prediction is of the same class as gt j
            self.gt_cls_matching = (pred_cls[:, None] == gt_cls[None, :])
            self.gt_cls_iou = self.gt_iou * self.gt_cls_matching

            # This will be changed in the matching calculation, so make a copy
            iou_buffer = self.gt_cls_iou.copy()
            iou_buffer2 = self.gt_iou.copy()

            for pred_idx, pred_elem in enumerate(preds):
                # Find the max iou ground truth for this prediction
                gt_idx = np.argmax(iou_buffer[pred_idx, :])
                iou = iou_buffer[pred_idx, gt_idx]

                pred_elem['iou'] = np.max(self.gt_cls_iou[pred_idx, :])

                if iou >= self.pos_thresh:
                    gt_elem = gt[gt_idx]

                    pred_elem['used'] = True
                    gt_elem['used'] = True
                    pred_elem['matched_with'] = gt_elem['_id']
                    gt_elem['matched_with'] = pred_elem['_id']

                    pred_elem['vis_gt_idx'] = gt_elem['_idx']

                    # Make sure this gt can't be used again
                    iou_buffer[:, gt_idx] = 0

                # match the most possibility gt for unmatched pred(visualized needed)
                if not pred_elem['used']:
                    if np.max(iou_buffer2[pred_idx, :]) == 0:
                        pred_elem['vis_gt_idx'] = None
                    else:
                        pred_elem['vis_gt_idx'] = gt[np.argmax(iou_buffer2[pred_idx, :])]['_idx']
                    pred_elem['iou'] = np.max(iou_buffer2[pred_idx, :])

        # Ignore regions annotations allow us to ignore predictions that fall within
        if len(ignore) > 0:
            # Because ignore regions have extra parameters, it's more efficient to use a for loop here
            for ignore_region in ignore:
                if ignore_region['mask'] is None and ignore_region['bbox'] is None:
                    # The region should span the whole image
                    ignore_iou = [1] * len(preds)
                else:
                    if ignore_region[det_type] is None:
                        # There is no det_type annotation for this specific region so skip it
                        continue
                    # Otherwise, compute the crowd IoU between the detections and this region

                    if self.isvideo:
                        ignore_iou = np.zeros((len(detections)))
                        for dind, di in enumerate(detections):
                            ignore_iou[dind] = iou_seq(di, ignore_region[det_type])
                    else:
                        ignore_iou = mask_utils.iou(detections, [ignore_region[det_type]], [True])

                for pred_idx, pred_elem in enumerate(preds):
                    if not pred_elem['used'] and (ignore_iou[pred_idx] > self.pos_thresh) \
                            and (ignore_region['class'] == pred_elem['class'] or ignore_region['class'] == -1):
                        # Set the prediction to be ignored
                        pred_elem['used'] = None

        # set ignore to ignored predict
        if self.isvideo:
            for idx, pred in enumerate(preds):
                if pred['used'] != None:
                    if pred['ignore'] and not pred['used']: pred['used'] = None
                    # if pred['ignore'] and pred['used']: print('yes')

        if len(gt) == 0:
            return

        # Some matrices used just for error calculation
        if self.run_errors:
            self.gt_used = np.array([x['used'] == True for x in gt])[None, :]
            self.gt_unused = ~self.gt_used

            self.gt_unused_iou = self.gt_unused * self.gt_iou
            self.gt_unused_cls = self.gt_unused_iou * self.gt_cls_matching
            self.gt_unused_noncls = self.gt_unused_iou * ~self.gt_cls_matching

            self.gt_noncls_iou = self.gt_iou * ~self.gt_cls_matching

            self.gt_used_iou = self.gt_used * self.gt_iou
            self.gt_used_cls = self.gt_used_iou * self.gt_cls_matching


class TIVERun(TIDERun):
    """ Holds the data for a single run of TIDE. """

    def __init__(self, gt: TiveData, preds: TiveData, pos_thresh: float, bg_thresh: float, mode: str, max_dets: int,
                 run_errors: bool = True, isvideo: bool = False, frame_thr: float = 0.1, temporal_thr: float = 0.4,
                 image_root: str = None):
        self.isvideo = isvideo
        self.temporal_thr = temporal_thr
        self.frame_thr = frame_thr

        self.image_root = image_root

        super().__init__(gt, preds, pos_thresh, bg_thresh, mode, max_dets, run_errors)

    def _run(self):
        """ And awaaay we go """

        for image in tqdm(self.gt.images, desc='evaluating thresh {}'.format(self.pos_thresh)):
            x = self.preds.get(image)
            y = self.gt.get(image)

            # These classes are ignored for the whole image and not in the ground truth, so
            # we can safely just remove these detections from the predictions at the start.
            # However, since ignored detections are still used for error calculations, we have to keep them.
            if not self.run_errors:
                ignored_classes = self.gt._get_ignored_classes(image)
                x = [pred for pred in x if pred['class'] not in ignored_classes]

            self._eval_image(x, y, image)

        # Store a fixed version of all the errors for testing purposes
        for error in self.errors:
            error.original = f.nonepack(error.unfix())
            error.fixed = f.nonepack(error.fix())
            error.disabled = False

        self.ap = self.ap_data.get_mAP()

        # Now that we've stored the fixed errors, we can clear the gt info
        self._clear()

    def _eval_image(self, preds: list, gt: list, image: int):

        for truth in gt:
            if not truth['ignore']:
                self.ap_data.add_gt_positives(truth['class'], 1)

        if len(preds) == 0:
            # There are no predictions for this image so add all gt as missed
            for truth in gt:
                if not truth['ignore']:
                    self.ap_data.push_false_negative(truth['class'], truth['_id'])

                    if self.run_errors:
                        self._add_error(MissedError(truth))
                        self.false_negatives[truth['class']].append(truth)
            return

        ex = TIVEExample(preds, gt, self.pos_thresh, self.mode, self.max_dets, self.run_errors, self.isvideo)
        preds = ex.preds  # In case the number of predictions was restricted to the max

        visualizer = Visualizer(ex, image, self.gt.images[image]['name'], self.image_root,
                                # save_root='./visualize_output')
                                save_root=r'E:\AAAAAAAAAAAAAAAAA\visualize_mtr')

        for pred_idx, pred in enumerate(preds):

            pred['info'] = {'iou': pred['iou'], 'used': pred['used']}
            if pred['used']:
                pred['info']['matched_with'] = pred['matched_with']
                if self.run_errors:
                    visualizer.draw(pred, 'TP')

            if pred['used'] is not None:
                self.ap_data.push(pred['class'], pred['_id'], pred['score'], pred['used'], pred['info'])

            # ----- ERROR DETECTION ------ #
            # This prediction is a negative (or ignored), let's find out why
            if self.run_errors and (pred['used'] == False or pred['used'] == None):
                # Test for BackgroundError
                if len(ex.gt) == 0:  # Note this is ex.gt because it doesn't include ignore annotations
                    # There is no ground truth for this image, so just mark everything as BackgroundError
                    self._add_error(BackgroundError(pred))
                    visualizer.draw(pred, BackgroundError.short_name)
                    continue

                # errors only for video
                if self.isvideo:

                    idx = ex.gt_cls_iou[pred_idx, :].argmax()
                    if self.bg_thresh <= ex.gt_cls_iou[pred_idx, idx] <= self.pos_thresh:
                        # calucate per frame iou

                        gt_pred = ex.gt[idx]

                        # gt_pred dict_keys(['_id', 'score', 'image', 'class', 'bbox', 'mask', 'ignore', 'used', 'usable', '_idx',
                        #           'matched_with'])
                        # pred dict_keys(
                        #    ['_id', 'score', 'image', 'class', 'bbox', 'mask', 'ignore', 'used', '_idx', 'iou', 'info'])

                        frame_gt_iou = np.zeros(len(pred['mask']))
                        gt_len = pr_len = 0
                        for _i, (_pr, _prgt) in enumerate(zip(pred['mask'], gt_pred['mask'])):
                            if _pr != None:
                                pr_mask = np.any(mask_utils.decode(_pr))
                            else:
                                pr_mask = False

                            if _prgt == None and not pr_mask:
                                # gt and pred both have no mask
                                tmp_fiou = 0.0
                            elif _prgt == None and pr_mask:
                                # gt has no mask and pred has mask
                                tmp_fiou = 0.0
                                pr_len += 1
                            elif _prgt != None and not pr_mask:
                                # gt has mask and pred has no mask
                                tmp_fiou = 0.0
                                gt_len += 1
                            else:
                                # gt and prd both have mask
                                tmp_fiou = mask_utils.iou([_pr], [_prgt], [False])
                                gt_len += 1
                                pr_len += 1

                            frame_gt_iou[_i] = tmp_fiou
                        temporal_good = 0
                        for _iou in frame_gt_iou:
                            if _iou > self.frame_thr:
                                temporal_good += 1

                        temporal_overlap = temporal_good / (gt_len + pr_len)

                        # Test for SpatialBadError
                        # This detection would have been positive if it had higher IoU with this GT
                        if temporal_overlap >= self.temporal_thr:
                            self._add_error(SpatialBadError(pred, ex.gt[idx], ex))
                            visualizer.draw(pred, SpatialBadError.short_name)

                            continue

                        # Test for TemporalBadError

                        # This detection would have been positive if it had higher IoU with this GT
                        # if temporal_overlap < self.temporal_thr:
                        else:
                            self._add_error(TemporalBadError(pred, ex.gt[idx], ex))
                            visualizer.draw(pred, TemporalBadError.short_name)

                            continue

                # Test for ClassError
                idx = ex.gt_noncls_iou[pred_idx, :].argmax()
                if ex.gt_noncls_iou[pred_idx, idx] >= self.pos_thresh:
                    # This detection would have been a positive if it was the correct class
                    self._add_error(ClassError(pred, ex.gt[idx], ex))
                    visualizer.draw(pred, ClassError.short_name)
                    continue

                # Test for DuplicateError
                idx = ex.gt_used_cls[pred_idx, :].argmax()
                if ex.gt_used_cls[pred_idx, idx] >= self.pos_thresh:
                    # The detection would have been marked positive but the GT was already in use
                    suppressor = self.preds.annotations[ex.gt[idx]['matched_with']]
                    self._add_error(DuplicateError(pred, suppressor))
                    visualizer.draw(pred, DuplicateError.short_name)
                    continue

                # Test for BackgroundError
                idx = ex.gt_iou[pred_idx, :].argmax()
                if ex.gt_iou[pred_idx, idx] <= self.bg_thresh:
                    # This should have been marked as background
                    self._add_error(BackgroundError(pred))
                    visualizer.draw(pred, BackgroundError.short_name)
                    continue

                # errors only for image
                if not self.isvideo:
                    # Test for BoxError
                    idx = ex.gt_cls_iou[pred_idx, :].argmax()
                    if self.bg_thresh <= ex.gt_cls_iou[pred_idx, idx] <= self.pos_thresh:
                        # This detection would have been positive if it had higher IoU with this GT
                        self._add_error(BoxError(pred, ex.gt[idx], ex))
                        continue

                    # A base case to catch uncaught errors
                    self._add_error(OtherError(pred))
                else:
                    self._add_error(VideoOtherError(pred))
                    visualizer.draw(pred, VideoOtherError.short_name)
        for truth in gt:
            # If the GT wasn't used in matching, meaning it's some kind of false negative
            if not truth['ignore'] and not truth['used']:
                self.ap_data.push_false_negative(truth['class'], truth['_id'])

                if self.run_errors:
                    self.false_negatives[truth['class']].append(truth)

                    # The GT was completely missed, no error can correct it
                    # Note: 'usable' is set in error.py
                    if not truth['usable']:
                        self._add_error(MissedError(truth))
                        visualizer.draw(truth['_idx'], MissedError.short_name)


class TIVE(TIDE):
    """


    ████████╗██╗██╗   ██╗███████╗
    ╚══██╔══╝██║██║   ██║██╔════╝
       ██║   ██║██║   ██║█████╗
       ██║   ██║╚██╗ ██╔╝██╔══╝
       ██║   ██║ ╚████╔╝ ███████╗
       ╚═╝   ╚═╝  ╚═══╝  ╚══════╝



   """

    # This is just here to define a consistent order of the error types

    _error_types_video = [ClassError, DuplicateError, SpatialBadError, TemporalBadError, VideoOtherError,BackgroundError,
                          MissedError]
    _error_types = [ClassError, BoxError, OtherError, DuplicateError, BackgroundError, MissedError]
    _special_error_types = [FalsePositiveError, FalseNegativeError]

    # Threshold splits for different challenges
    COCO_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    VOL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Threshold splits for different length for sequence
    SEQ_THRESHOLDS = [16, 32]  # [16, 32]

    # The modes of evaluation
    BOX = 'bbox'
    MASK = 'mask'

    def __init__(self, pos_threshold: float = 0.5, background_threshold: float = 0.1, mode: str = MASK,
                 isvideo: bool = False, frame_thr: float = 0.1, temporal_thr: float = 0.4,
                 image_root: str = None):
        super().__init__(pos_threshold, background_threshold, mode)
        self.isvideo = isvideo
        self.temporal_thr = temporal_thr
        self.frame_thr = frame_thr
        # if image root not None ,save the visualize output
        self.image_root = image_root

        if self.isvideo:
            TIVE._error_types = TIVE._error_types_video
            TIDE._error_types = TIVE._error_types_video

        self.plotter = P.TivePlotter(isvideo=self.isvideo)

    def evaluate(self, gt: TiveData, preds: TiveData, pos_threshold: float = None, background_threshold: float = None,
                 mode: str = None, name: str = None, use_for_errors: bool = True) -> TIDERun:
        pos_thresh = self.pos_thresh if pos_threshold is None else pos_threshold
        bg_thresh = self.bg_thresh if background_threshold is None else background_threshold
        mode = self.mode if mode is None else mode
        name = preds.name if name is None else name

        run = TIVERun(gt, preds, pos_thresh, bg_thresh, mode, gt.max_dets, use_for_errors,
                      self.isvideo, self.frame_thr, self.temporal_thr, self.image_root)

        if use_for_errors:
            self.runs[name] = run

        return run

    def evaluate_all(self, gt: TiveData, preds: TiveData, seq_thresholds: list = SEQ_THRESHOLDS,
                     thresholds: list = COCO_THRESHOLDS, pos_threshold: float = None,
                     background_threshold: float = None, mode: str = None, name: str = None) -> dict:
        gt_short, gt_medium, gt_long = self.divide_sequence(gt, seq_thresholds)
        preds_short, preds_medium, preds_long = self.divide_sequence(preds, seq_thresholds)

        # evaluate
        # first evaluate all gts and detections
        print('=' * 40 + 'evaluating all gts and detections' + '=' * 40)
        self.evaluate_range(gt, preds, thresholds, pos_threshold, background_threshold, mode, name)
        # evaluate on short, medium, long
        print('=' * 40 + 'evaluating short sequences' + '=' * 40)
        self.evaluate_range(gt_short, preds_short, thresholds, pos_threshold, background_threshold, mode, 'short')
        print('=' * 40 + 'evaluating medium sequences' + '=' * 40)
        self.evaluate_range(gt_medium, preds_medium, thresholds, pos_threshold, background_threshold, mode, 'medium')
        print('=' * 40 + 'evaluating long sequences' + '=' * 40)
        self.evaluate_range(gt_long, preds_long, thresholds, pos_threshold, background_threshold, mode, 'long')

    def summarize(self):
        """ Summarizes the mAP values and errors for all runs in this TIDE object. Results are printed to the console. """
        main_errors = self.get_main_errors()
        special_errors = self.get_special_errors()

        for run_name, run in self.runs.items():
            print('-- {} --\n'.format(run_name))

            # If we evaluated on all thresholds, print them here
            if run_name in self.run_thresholds:
                thresh_runs = self.run_thresholds[run_name]
                aps = [trun.ap for trun in thresh_runs]

                # Print Overall AP for a threshold run
                ap_title = '{} AP @ [{:d}-{:d}]'.format(thresh_runs[0].mode,
                                                        int(thresh_runs[0].pos_thresh * 100),
                                                        int(thresh_runs[-1].pos_thresh * 100))
                print('{:s}: {:.2f}'.format(ap_title, sum(aps) / len(aps)))

                # Print AP for every threshold on a threshold run
                P.print_table([
                    ['Thresh'] + [str(int(trun.pos_thresh * 100)) for trun in thresh_runs],
                    ['  AP  '] + ['{:6.2f}'.format(trun.ap) for trun in thresh_runs]
                ], title=ap_title)

                # Print qualifiers for a threshold run
                if len(self.qualifiers) > 0:
                    print()
                    # Can someone ban me from using list comprehension? this is unreadable
                    qAPs = [
                        f.mean(
                            [trun.qualifiers[q] for trun in thresh_runs if q in trun.qualifiers]
                        ) for q in self.qualifiers
                    ]

                    P.print_table([
                        ['Name'] + list(self.qualifiers.keys()),
                        [' AP '] + ['{:6.2f}'.format(qAP) for qAP in qAPs]
                    ], title='Qualifiers {}'.format(ap_title))

            # Otherwise, print just the one run we did
            else:
                # Print Overall AP for a regular run
                ap_title = '{} AP @ {:d}'.format(run.mode, int(run.pos_thresh * 100))
                print('{}: {:.2f}'.format(ap_title, run.ap))

                # Print qualifiers for a regular run
                if len(self.qualifiers) > 0:
                    print()
                    qAPs = [run.qualifiers[q] if q in run.qualifiers else 0 for q in self.qualifiers]
                    P.print_table([
                        ['Name'] + list(self.qualifiers.keys()),
                        [' AP '] + ['{:6.2f}'.format(qAP) for qAP in qAPs]
                    ], title='Qualifiers {}'.format(ap_title))

            print()
            # Print the main errors
            P.print_table([
                ['Type'] + [err.short_name for err in TIDE._error_types],
                [' dAP'] + ['{:6.2f}'.format(main_errors[run_name][err.short_name]) for err in TIDE._error_types]
            ], title='Main Errors')

            print()
            # Print the special errors
            P.print_table([
                ['Type'] + [err.short_name for err in TIDE._special_error_types],
                [' dAP'] + ['{:6.2f}'.format(special_errors[run_name][err.short_name]) for err in
                            TIDE._special_error_types]
            ], title='Special Error')

            print()

    def plot(self, out_dir: str = None):
        """
        Plots a summary model for each run in this TIDE object.
        Images will be outputted to out_dir, which will be created if it doesn't exist.
        """

        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        errors = self.get_all_errors()

        if len(errors) == 0:
            return

        max_main_error = max(sum([list(x.values()) for x in errors['main'].values()], []))
        max_spec_error = max(sum([list(x.values()) for x in errors['special'].values()], []))
        dap_granularity = 5  # The max will round up to the nearest unit of this

        # Round the plotter's dAP range up to the nearest granularity units
        if max_main_error > self.plotter.MAX_MAIN_DELTA_AP:
            self.plotter.MAX_MAIN_DELTA_AP = math.ceil(max_main_error / dap_granularity) * dap_granularity
        if max_spec_error > self.plotter.MAX_SPECIAL_DELTA_AP:
            self.plotter.MAX_SPECIAL_DELTA_AP = math.ceil(max_spec_error / dap_granularity) * dap_granularity

        # Do the plotting now
        for run_name, run in self.runs.items():
            self.plotter.make_summary_plot(out_dir, errors, run_name, run.mode, hbar_names=True)


    def divide_sequence(self, data_in: TiveData, seq_thresholds):
        data_short, data_medium, data_long = TiveData('short'), TiveData('medium'), TiveData('long')

        # divide annos or detections into short, medium, long
        for im_id in data_in.images:
            annos = data_in.get(im_id)
            for _a in annos:
                if _a['gt_length'] <= seq_thresholds[0]:
                    if im_id not in data_short.images:
                        data_short.add_image(im_id, data_in.images[im_id]['name'])
                    data_short._add(im_id, _a['class'], _a['bbox'], _a['mask'], _a['score'], _a['ignore'],
                                    _a['gt_length'])
                else:
                    data_short._add(im_id, _a['class'], _a['bbox'], _a['mask'], _a['score'], True, _a['gt_length'])

                if seq_thresholds[1] >= _a['gt_length'] > seq_thresholds[0]:
                    if im_id not in data_medium.images:
                        data_medium.add_image(im_id, data_in.images[im_id]['name'])
                    data_medium._add(im_id, _a['class'], _a['bbox'], _a['mask'], _a['score'], _a['ignore'],
                                     _a['gt_length'])
                else:
                    data_medium._add(im_id, _a['class'], _a['bbox'], _a['mask'], _a['score'], True, _a['gt_length'])

                if _a['gt_length'] > seq_thresholds[1]:
                    if im_id not in data_long.images:
                        data_long.add_image(im_id, data_in.images[im_id]['name'])
                    data_long._add(im_id, _a['class'], _a['bbox'], _a['mask'], _a['score'], _a['ignore'],
                                   _a['gt_length'])
                else:
                    data_long._add(im_id, _a['class'], _a['bbox'], _a['mask'], _a['score'], True, _a['gt_length'])

        return data_short, data_medium, data_long
