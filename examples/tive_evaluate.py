import sys
import argparse

sys.path.append("..")
from tivecv import TIVE
import tivecv.datasets as datasets

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, default='path/to/your/prediction/file/results.json', help='video root directory')
parser.add_argument('--gt_path', type=str, default='path/to/your/annoattion/file/annotations.json', help='video root directory')
parser.add_argument('--dataset_name', type=str, default='ytvis', help='video root directory')
args = parser.parse_args()

# load ytvis style annotation and result json files
gt = datasets.VideoData(path=args.gt_path)
mask_results = datasets.VideoDataResult(path=args.results_path)

# if you want to visualize the predictions,
# set image_root = 'root/to/images' and visualize_root='/path/to/store/visualize_image'
tive = TIVE(isvideo=True, image_root=None, visualize_root=None, dataset_name=args.dataset_name)

tive.evaluate_all(gt, mask_results, mode=TIVE.MASK)
tive.summarize()
tive.plot(out_dir='./tive_output')
