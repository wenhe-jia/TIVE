# A General **T**oolbox for **I**dentifying **V**ideo Instance Segmentation **E**rrors

By **Wenhe Jia**, **Lu Yang**, **Zilong Jia**, **Qing Song**

```
████████╗██╗██╗   ██╗███████╗
╚══██╔══╝██║██║   ██║██╔════╝
   ██║   ██║██║   ██║█████╗
   ██║   ██║╚██╗ ██╔╝██╔══╝
   ██║   ██║ ╚████╔╝ ███████╗
   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝
```

An easy-to-use, general toolbox to compute and evaluate the effect of video instance segmentation on overall performance. 

[ArXiv]()(need to be updated).

## Installation

TIVE is available as a python package for python 3.6+, based on [TIDE](https://github.com/dbolya/tide), we reimplemented specific modules for video instance segmentation. To get started with TIVE, simply install TIDE first with pip:

```shell
pip3 install tidecv
```


## Datasets

The currently supported YouTube-VIS dataset. json file for YouTubeVIS-2021 `mini_train` and `minival` can be found in [YouTubeVIS-2021-minitrain/minival]()(need to be updated). To evaluate on other common VIS datasets, you need to convert your dataset's format same as YouTube-VIS.


## Get Started

TIVE is meant as a drop-in replacement for the [YouTubeVIS Evaluation toolkit]([https://github.com/youtubevos/cocoapi]), get detailed evaluation results on YoutubeVIS-2021-minival subset:

```python
from tivecv import TIVE
import tivecv.datasets as datasets

image_root = 'path/to/youtubevis_2021_train/images'
gt = datasets.YTVIS2021(path='path/to/youtubevis_2021_minival.json')
result = datasets.YTVIS2021Result('path/to/your/results/file')

tive = TIVE(isvideo=True, image_root=image_root)
tive.evaluate_all(gt, result, mode=TIVE.MASK)

tive.summarize()
tive.plot(out_dir='./tive_output')
```

This prints evaluation summary tables to the console:
```

(need to be updated)

```

And a summary plot for your model's errors:

![A summary plot](https://github.com/wenhe-jia/TIVE/blob/main/examples/results_sequence_mask_summary.png)


## Citation

If you use TIVE in your project, please cite
```
@inproceedings{jia2022tive,
  author    = {Wenhe Jia and Lu Yang and Zilong Jia and Qing Song},
  title     = {TIVE: A General Toolbox for Identifying Video Instance Segemntation Errors},
  booktitle = {arXiv},
  year      = {2022},
}
```


## Acknowledgement

Code is largely based on TIDE (https://github.com/dbolya/tide).
