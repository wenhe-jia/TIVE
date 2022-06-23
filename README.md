# A General **T**oolbox for **I**dentifying **V**ideo Instance Segmentation **E**rrors

```
████████╗██╗██╗   ██╗███████╗
╚══██╔══╝██║██║   ██║██╔════╝
   ██║   ██║██║   ██║█████╗
   ██║   ██║╚██╗ ██╔╝██╔══╝
   ██║   ██║ ╚████╔╝ ███████╗
   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝
```

An easy-to-use, general toolbox to compute and evaluate the effect of video instance segmentation on overall performance. 
([ArXiv]())need to be updated .

# Installation

TIVE is available as a python package for python 3.6+, based on ([TIDE](https://github.com/dbolya/tide)). To install TIVE, simply install TIDE first  with pip:

```shell
pip3 install tidecv
```

# Usage
TIDE is meant as a drop-in replacement for the [YouTubeVIS Evaluation toolkit]([https://github.com/youtubevos/cocoapi]), get detailed evaluation results on YoutubeVIS-2021-minival subset:

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

need to be updated

```

And a summary plot for your model's errors:

![A summary plot](https://dbolya.github.io/tide/mask_rcnn_bbox_bbox_summary.png)

## Jupyter Notebook

Check out the [example notebook](https://github.com/dbolya/tide/blob/master/examples/coco_instance_segmentation.ipynb) for more details.


# Datasets
The currently supported datasets are COCO, LVIS, Pascal, and Cityscapes. More details and documentation on how to write your own database drivers coming soon!

# Citation
If you use TIDE in your project, please cite
```
@inproceedings{tide-eccv2020,
  author    = {Daniel Bolya and Sean Foley and James Hays and Judy Hoffman},
  title     = {TIDE: A General Toolbox for Identifying Object Detection Errors},
  booktitle = {ECCV},
  year      = {2020},
}
```

## Contact
For questions about our paper or code, make an issue in this github or contact [Daniel Bolya](mailto:dbolya@gatech.edu). Note that I may not respond to emails, so github issues are your best bet.
