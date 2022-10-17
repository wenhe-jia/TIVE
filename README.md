# A **T**oolbox for **I**dentifying **V**ideo Instance Segmentation **E**rrors

<img src="./examples/tive_logo.svg" width="100px">

> [A Toolbox for Identifying Video Instance Segmentation Errors]() <br>
> [![paper](https://img.shields.io/badge/Paper-arxiv-b31b1b)](https://)

An toolbox to evaluate the effect of video instance segmentation errors on overall performance. 

## Installation

TIVE is available as a python package for python 3.6+, based on [TIDE](https://github.com/dbolya/tide), we reimplemented specific modules for video instance segmentation. To get started with TIVE, simply install TIDE first with pip:

```shell
pip3 install tidecv
```


## Datasets

The currently supported YouTube-VIS dataset. json file for YouTubeVIS-2021 `mini_train` and `minival` can be found in [YouTubeVIS-2021-minitrain/minival](https://pan.baidu.com/s/1EFgzjxRTLa4c13izEVkFNQ?pwd=e6kj)(code: e6kj). To evaluate on other common VIS datasets, you need to convert your dataset's format same as YouTube-VIS.


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

-- results_seq_r50 --

mask AP @ [50-95]: 43.37
                                         mask AP @ [50-95]
===================================================================================================
  Thresh       50       55       60       65       70       75       80       85       90       95  
---------------------------------------------------------------------------------------------------
    AP      61.03    59.22    57.56    55.47    52.87    47.91    41.99    33.13    21.56     2.95  
===================================================================================================

                             Main Errors
======================================================================
  Type      Cls     Dupe     Spat     Temp     Both      Bkg     Miss  
----------------------------------------------------------------------
   dAP     6.95     0.25     7.69     4.00     0.00     0.97     6.20  
======================================================================

        Special Error
=============================
  Type   FalsePos   FalseNeg  
-----------------------------
   dAP       5.90      16.49  
=============================

-- short --

mask AP @ [50-95]: 21.69
                                         mask AP @ [50-95]
===================================================================================================
  Thresh       50       55       60       65       70       75       80       85       90       95  
---------------------------------------------------------------------------------------------------
    AP      39.44    34.00    31.59    25.46    24.70    19.94    19.24    17.68     4.82     0.06  
===================================================================================================

                             Main Errors
======================================================================
  Type      Cls     Dupe     Spat     Temp     Both      Bkg     Miss  
----------------------------------------------------------------------
   dAP     1.97     0.00     4.76    20.43     0.00     3.41     9.69  
======================================================================

        Special Error
=============================
  Type   FalsePos   FalseNeg  
-----------------------------
   dAP       2.57      24.15  
=============================

-- medium --

mask AP @ [50-95]: 40.43
                                         mask AP @ [50-95]
===================================================================================================
  Thresh       50       55       60       65       70       75       80       85       90       95  
---------------------------------------------------------------------------------------------------
    AP      54.69    52.87    51.42    50.79    49.11    44.09    38.63    30.22    24.97     7.47  
===================================================================================================

                             Main Errors
======================================================================
  Type      Cls     Dupe     Spat     Temp     Both      Bkg     Miss  
----------------------------------------------------------------------
   dAP     6.40     0.04     9.62     5.72     0.00     0.67     4.64  
======================================================================

        Special Error
=============================
  Type   FalsePos   FalseNeg  
-----------------------------
   dAP       3.77      13.79  
=============================

-- long --

mask AP @ [50-95]: 39.20
                                         mask AP @ [50-95]
===================================================================================================
  Thresh       50       55       60       65       70       75       80       85       90       95  
---------------------------------------------------------------------------------------------------
    AP      56.45    54.79    53.68    49.80    45.59    42.41    38.45    30.46    18.10     2.25  
===================================================================================================

                             Main Errors
======================================================================
  Type      Cls     Dupe     Spat     Temp     Both      Bkg     Miss  
----------------------------------------------------------------------
   dAP    15.99     0.50     4.82     3.03     1.45     1.81     3.96  
======================================================================

        Special Error
=============================
  Type   FalsePos   FalseNeg  
-----------------------------
   dAP       4.44       8.18  
=============================

```

And a summary plot for your model's errors:

![A summary plot](./examples/results_sequence_mask_summary.png)


## Citation

If you use TIVE in your project, please cite
```
@inproceedings{
    TO Be Updated
}
```


## Acknowledgement

Code is largely based on TIDE (https://github.com/dbolya/tide).
