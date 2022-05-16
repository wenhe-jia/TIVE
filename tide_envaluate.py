# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File ：tide_envaluate.py
@Author ：jzl
@Date ：2022/3/13 14:39 
'''

from tidecv import TIDE
import tidecv.datasets as datasets

# gt = datasets.YTVIS2021_perimg(path='/home/jwh/vis/mini360relate/valid_mini.json')
# mask_results = datasets.YTVIS2021Result_perimg(path='/home/jwh/vis/mini360relate/results_minioriginal.json',
#                                                data_ann=gt)
# tide = TIDE()

gt = datasets.YTVIS2021(path='G:/code/ytvis2022/mini360relate/valid_mini.json')
# mask_results = datasets.YTVIS2021Result(path=r'G:\code\ytvis2022\mini360relate\results_seq_r50.json', )
mask_results = datasets.YTVIS2021Result(path=r'G:\code\ytvis2022\mini360relate\results_mtr_r50.json', )



# image_root = r'E:\BaiduNetdiskDownload\vis_datasets\vis_2021\train\JPEGImages'
image_root = None
tide = TIDE(isvideo=True, image_root=image_root)

# tide.evaluate_range(gt, mask_results, mode=TIDE.MASK)
tide.evaluate_length(gt, mask_results, mode=TIDE.MASK)

tide.summarize()
# tide.plot(out_dir='./tide_output')
