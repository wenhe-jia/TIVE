# Modified by Zilong Jia from https://github.com/dbolya/tide
import math
import sys
from collections import defaultdict, OrderedDict
import os
import shutil

import cv2
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
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
            # self.colors_main = OrderedDict({
            #     ClassError.short_name: current_palette[9],
            #     DuplicateError.short_name: current_palette[6],
            #     SpatialBadError.short_name: current_palette[8],
            #     TemporalBadError.short_name: current_palette[2],
            #     VideoOtherError.short_name: current_palette[7],
            #     BackgroundError.short_name: current_palette[4],
            #     MissedError.short_name: current_palette[3],
            #
            # })

            self.colors_main = OrderedDict({
                ClassError.short_name: '#FACE32',
                DuplicateError.short_name: '#9670BA',
                SpatialBadError.short_name: '#80BA0E',
                TemporalBadError.short_name: '#E75D6D',
                VideoOtherError.short_name: '#019E97',
                BackgroundError.short_name: '#FDA632',
                MissedError.short_name: '#6796F4',
            })

    def make_summary_plot(self, out_dir: str, errors: dict, model_name: str, rec_type: str, hbar_names: bool = False):
        """Make a summary plot of the errors for a model, and save it to the figs folder.

        :param out_dir:    The output directory for the summary image. MUST EXIST.
        :param errors:     Dictionary of both main and special errors.
        :param model_name: Name of the model for which to generate the plot.
        :param rec_type:   Recognition type, either TIDE.BOX or TIDE.MASK
        :param hbar_names: Whether or not to include labels for the horizontal bars.
        """

        tmp_dir = self._prepare_tmp_dir()

        high_dpi = int(500 * self.quality)
        low_dpi = int(300 * self.quality)

        # get the data frame
        error_dfs = {errtype: pd.DataFrame(data={
            'Error Type': list(errors[errtype][model_name].keys()),
            'Delta mAP': list(errors[errtype][model_name].values()),
        }) for errtype in ['main', 'special']}

        # pie plot for error type breakdown
        error_types = list(errors['main'][model_name].keys()) + list(errors['special'][model_name].keys())

        error_sum = sum([e for e in errors['main'][model_name].values()])
        error_sizes = [e / error_sum for e in errors['main'][model_name].values()] + [0, 0]
        fig, ax = plt.subplots(1, 1, figsize=(11, 11), dpi=high_dpi)
        patches, outer_text, inner_text = ax.pie(error_sizes, colors=self.colors_main.values(), labels=error_types,
                                                 autopct='%1.1f%%', startangle=90)
        for text in outer_text + inner_text:
            text.set_text('')
        for i in range(len(self.colors_main)):
            if error_sizes[i] > 0.05:
                inner_text[i].set_text(list(self.colors_main.keys())[i])
            inner_text[i].set_fontsize(48)
            inner_text[i].set_fontweight('bold')
        ax.axis('equal')
        plt.title(model_name, fontdict={'fontsize': 60, 'fontweight': 'bold'})
        pie_path = os.path.join(tmp_dir, '{}_{}_pie.png'.format(model_name, rec_type))
        plt.savefig(pie_path, bbox_inches='tight', dpi=low_dpi)
        plt.close()

        # horizontal bar plot for main error types
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=high_dpi)
        sns.barplot(data=error_dfs['main'], x='Delta mAP', y='Error Type', ax=ax,
                    palette=self.colors_main.values())

        ax.xaxis.set_major_locator(MultipleLocator(math.ceil(self.MAX_MAIN_DELTA_AP / 5)))
        ax.set_xlim(0, self.MAX_MAIN_DELTA_AP, auto=True)
        ax.set_xlabel('')
        ax.set_ylabel('')

        if not hbar_names:
            ax.set_yticklabels([''] * 6)
        plt.setp(ax.get_xticklabels(), fontsize=28)
        plt.setp(ax.get_yticklabels(), fontsize=36)
        ax.grid(False)
        sns.despine(left=True, bottom=True, right=True)
        hbar_path = os.path.join(tmp_dir, '{}_{}_hbar.png'.format(model_name, rec_type))
        plt.savefig(hbar_path, bbox_inches='tight', dpi=low_dpi)
        plt.close()

        # # vertical bar plot for special error types
        # fig, ax = plt.subplots(1, 1, figsize=(2, 5), dpi=high_dpi)
        # sns.barplot(data=error_dfs['special'], x='Error Type', y='Delta mAP', ax=ax,
        #             palette=self.colors_special.values())
        # ax.set_ylim(0, self.MAX_SPECIAL_DELTA_AP)
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # ax.set_xticklabels(['FP', 'FN'])
        # plt.setp(ax.get_xticklabels(), fontsize=36)
        # plt.setp(ax.get_yticklabels(), fontsize=28)
        # ax.grid(False)
        # sns.despine(left=True, bottom=True, right=True)
        # vbar_path = os.path.join(tmp_dir, '{}_{}_vbar.png'.format(model_name, rec_type))
        # plt.savefig(vbar_path, bbox_inches='tight', dpi=low_dpi)
        # plt.close()

        # get each subplot image
        pie_im = cv2.imread(pie_path)
        hbar_im = cv2.imread(hbar_path)
        # vbar_im = cv2.imread(vbar_path)

        # pad the hbar image vertically
        # summary_im = np.concatenate([np.zeros((vbar_im.shape[0] - hbar_im.shape[0], hbar_im.shape[1], 3)) + 255, hbar_im],
        #                          axis=0)
        # summary_im = np.concatenate([hbar_im, vbar_im], axis=1)
        summary_im = hbar_im

        # pad summary_im
        if summary_im.shape[1] < pie_im.shape[1]:
            lpad, rpad = int(np.ceil((pie_im.shape[1] - summary_im.shape[1]) / 2)), \
                         int(np.floor((pie_im.shape[1] - summary_im.shape[1]) / 2))
            summary_im = np.concatenate([np.zeros((summary_im.shape[0], lpad, 3)) + 255,
                                         summary_im,
                                         np.zeros((summary_im.shape[0], rpad, 3)) + 255], axis=1)

        # pad pie_im
        else:
            lpad, rpad = int(np.ceil((summary_im.shape[1] - pie_im.shape[1]) / 2)), \
                         int(np.floor((summary_im.shape[1] - pie_im.shape[1]) / 2))
            pie_im = np.concatenate([np.zeros((pie_im.shape[0], lpad, 3)) + 255,
                                     pie_im,
                                     np.zeros((pie_im.shape[0], rpad, 3)) + 255], axis=1)

        summary_im = np.concatenate([pie_im, summary_im], axis=0)

        if out_dir is None:
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow((summary_im / 255)[:, :, (2, 1, 0)])
            plt.show()
            plt.close()
        else:
            cv2.imwrite(os.path.join(out_dir, '{}_{}_summary.png'.format(model_name, rec_type)), summary_im)
