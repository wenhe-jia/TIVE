# Modified by Zilong Jia from https://github.com/dbolya/tide
import sys
from collections import defaultdict, OrderedDict
import os
import shutil

import cv2
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from tidecv.errors.main_errors import *
from .errors.main_errors import *
from tidecv.datasets import get_tide_path
from tidecv.plotting import print_table, Plotter


class TivePlotter(Plotter):
    """ Sets up a seaborn environment and holds the functions for plotting our figures. """

    def __init__(self, quality: float = 1, isvideo: bool = False):
        super().__init__(quality)
        self.isvideo = isvideo

        # Set mpl DPI in case we want to output to the screen / notebook
        mpl.rcParams['figure.dpi'] = 150

        # Seaborn color palette
        sns.set_palette('muted', 10)
        current_palette = sns.color_palette()

        # Seaborn style
        sns.set(style="whitegrid")
        
        if self.isvideo:
            self.colors_main = OrderedDict({
                ClassError.short_name: current_palette[9],
                DuplicateError.short_name: current_palette[6],
                SpatialBadError.short_name: current_palette[8],
                TemporalBadError.short_name: current_palette[2],
                BackgroundError.short_name: current_palette[4],
                MissedError.short_name: current_palette[3],
                VideoOtherError.short_name: current_palette[7],
            })

