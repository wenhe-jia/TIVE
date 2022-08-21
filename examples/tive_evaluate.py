import sys

sys.path.append("..")
from tivecv import TIVE
import tivecv.datasets as datasets

gt = datasets.YTVIS2021(path='path/to/annotation')

mask_results = datasets.YTVIS2021Result(path='path/to/result')

# image_root = 'root/to/images'
# if image_root is None ,we won't store the vislualize results
image_root = None
tive = TIVE(isvideo=True, image_root=image_root, visualize_root="/path/to/store/visualize_image")

# tide.evaluate_range(gt, mask_results, mode=TIDE.MASK)
tive.evaluate_all(gt, mask_results, mode=TIVE.MASK)

tive.summarize()
tive.plot(out_dir='./tive_output')
