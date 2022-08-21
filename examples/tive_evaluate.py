import sys

sys.path.append("..")
from tivecv import TIVE
import tivecv.datasets as datasets

# load ytvis style annotation and result json files
gt = datasets.YTVIS2021(path='path/to/annotation')
mask_results = datasets.YTVIS2021Result(path='path/to/result')

# if you want to visualize the predictions,
# set image_root = 'root/to/images' and visualize_root='/path/to/store/visualize_image'
tive = TIVE(isvideo=True, image_root=None, visualize_root=None)

tive.evaluate_all(gt, mask_results, mode=TIVE.MASK, seq_range=TIVE.YTVIS_SEQ_RANGE)
tive.summarize()
tive.plot(out_dir='./tive_output')
