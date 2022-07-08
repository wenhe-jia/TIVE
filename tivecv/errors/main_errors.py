# Modified by Zilong Jia from https://github.com/dbolya/tide
from collections import defaultdict
import numpy as np

from tidecv.errors.error import Error, BestGTMatch


class SpatialBadError(Error):
    description = "Error caused when a prediction would have been marked positive if it was localized better."
    short_name = "Spat"

    def __init__(self, pred: dict, gt: dict,ex):
        self.pred = pred
        self.gt = gt

        self.match = BestGTMatch(pred, gt) if not self.gt['used'] else None

    def fix(self):
        if self.match is None:
            return None
        return self.pred['class'], self.match.fix()


class TemporalBadError(Error):
    description = "Error caused when a prediction would have been marked positive if it was localized better."
    short_name = "Temp"

    def __init__(self, pred: dict, gt: dict,ex):
        self.pred = pred
        self.gt = gt

        self.match = BestGTMatch(pred, gt) if not self.gt['used'] else None

    def fix(self):
        if self.match is None:
            return None
        return self.pred['class'], self.match.fix()


class VideoOtherError(Error):
    description = "This detection didn't fall into any of the other error categories."
    short_name = "Other"

    def __init__(self, pred: dict):
        self.pred = pred

    def fix(self):
        return None
