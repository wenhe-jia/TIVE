from tivecv import TIVE
import tivecv.datasets as datasets

# gt = datasets.YTVIS2021_perimg(path='/home/jwh/vis/mini360relate/valid_mini.json')
# mask_results = datasets.YTVIS2021Result_perimg(path='/home/jwh/vis/mini360relate/results_minioriginal.json',
#                                                data_ann=gt)
# tide = TIDE()

# gt = datasets.YTVIS2021(path='G:/code/ytvis2022/mini360relate/valid_mini.json')
gt = datasets.YTVIS2021(path='/home/user/Program/vis/ytvis2021_mini/valid_mini.json')

# mask_results = datasets.YTVIS2021Result(path=r'/home/user/Program/vis/Mask2Former/output/inference/results.json', )
mask_results = datasets.YTVIS2021Result(path=r'/home/user/Program/vis/Mask2Former/48.00_output_lr0.00005_iter16000_T=2_batch8/inference/results.json', )



# image_root = r'E:\BaiduNetdiskDownload\vis_datasets\vis_2021\train\JPEGImages'
image_root = None
tive = TIVE(isvideo=True, image_root=image_root)

# tide.evaluate_range(gt, mask_results, mode=TIDE.MASK)
tive.evaluate_all(gt, mask_results, mode=TIVE.MASK)

tive.summarize()
tive.plot(out_dir='./tive_output')
