# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File ：visualizer.py
@Author ：jzl
@Date ：2022/4/30 17:44 
'''
import copy
import os, sys
import cv2
import numpy as np
import pycocotools.mask as mask_utils

_ID_JITTERS = [[0.9047944201469568, 0.3241718265806123, 0.33443746665210006],
               [0.4590171386127151, 0.9095038146383864, 0.3143840671974788],
               [0.4769356899795538, 0.5044406738441948, 0.5354530846360839],
               [0.00820945625670777, 0.24099210193126785, 0.15471834055332978],
               [0.6195684374237388, 0.4020380013509799, 0.26100266066404676],
               [0.08281237756545068, 0.05900744492710419, 0.06106221202154216],
               [0.2264886829978755, 0.04925271007292076, 0.10214429345996079],
               [0.1888247470009874, 0.11275000298612425, 0.46112894830685514],
               [0.37415767691880975, 0.844284596118331, 0.950471611180866],
               [0.3817344218157631, 0.3483259270707101, 0.6572989333690541],
               [0.2403115731054466, 0.03078280287279167, 0.5385975692534737],
               [0.7035076951650824, 0.12352084932325424, 0.12873080308790197],
               [0.12607434914489934, 0.111244793010015, 0.09333334699716023],
               [0.6551607300342269, 0.7003064103554443, 0.4131794512286162],
               [0.13592107365596595, 0.5390702818232149, 0.004540643174930525],
               [0.38286244894454347, 0.709142545393449, 0.529074791609835],
               [0.4279376583651734, 0.5634708596431771, 0.8505569717104301],
               [0.3460488523902999, 0.464769595519293, 0.6676839675477276],
               [0.8544063246675081, 0.5041190233407755, 0.9081217697141578],
               [0.9207009090747208, 0.2403865944739051, 0.05375410999863772],
               [0.6515786136947107, 0.6299918449948327, 0.45292029442034387],
               [0.986174217295693, 0.2424849846977214, 0.3981993323108266],
               [0.22101915872994693, 0.3408589198278038, 0.006381420347677524],
               [0.3159785813515982, 0.1145748921741011, 0.595754317197274],
               [0.10263421488052715, 0.5864139253490858, 0.23908000741142432],
               [0.8272999391532938, 0.6123527260897751, 0.3365197327803193],
               [0.5269583712937912, 0.25668929554516506, 0.7888411215078127],
               [0.2433880265410031, 0.7240751234287827, 0.8483215810528648],
               [0.7254601709704898, 0.8316525547295984, 0.9325253855921963],
               [0.5574483824856672, 0.2935331727879944, 0.6594839453793155],
               [0.6209642371433579, 0.054030693198821256, 0.5080873988178534],
               [0.9055507077365624, 0.12865888619203514, 0.9309191861440005],
               [0.9914469722960537, 0.3074114506206205, 0.8762107657323488],
               [0.4812682518247371, 0.15055826298548158, 0.9656340505308308],
               [0.6459219454316445, 0.9144794010251625, 0.751338812155106],
               [0.860840174209798, 0.8844626353077639, 0.3604624506769899],
               [0.8194991672032272, 0.926399617787601, 0.8059222327343247],
               [0.6540413175393658, 0.04579445254618297, 0.26891917826531275],
               [0.37778835833987046, 0.36247927666109536, 0.7989799305827889],
               [0.22738304978177726, 0.9038018263773739, 0.6970838854138303],
               [0.6362015495896184, 0.527680794236961, 0.5570915425178721],
               [0.6436401915860954, 0.6316925317144524, 0.9137151236993912],
               [0.04161828388587163, 0.3832413349082706, 0.6880829921949752],
               [0.7768167825719299, 0.8933821497682587, 0.7221278391266809],
               [0.8632760876301346, 0.3278628094906323, 0.8421587587114462],
               [0.8556499133262127, 0.6497385872901932, 0.5436895688477963],
               [0.9861940318610894, 0.03562313777386272, 0.9183454677106616],
               [0.8042586091176366, 0.6167222703170994, 0.24181981557207644],
               [0.9504247117633057, 0.3454233714011461, 0.6883727005547743],
               [0.9611909135491202, 0.46384154263898114, 0.32700443315058914],
               [0.523542176970206, 0.446222414615845, 0.9067402987747814],
               [0.7536954008682911, 0.6675512338797588, 0.22538238957839196],
               [0.1554052265688285, 0.05746097492966129, 0.8580358872587424],
               [0.8540838640971405, 0.9165504335482566, 0.6806982829158964],
               [0.7065090319405029, 0.8683059983962002, 0.05167128320624026],
               [0.39134812961899124, 0.8910075505622979, 0.7639815712623922],
               [0.1578117311479783, 0.20047326898284668, 0.9220177338840568],
               [0.2017488993096358, 0.6949259970936679, 0.8729196864798128],
               [0.5591089340651949, 0.15576770423813258, 0.1469857469387812],
               [0.14510398622626974, 0.24451497734532168, 0.46574271993578786],
               [0.13286397822351492, 0.4178244533944635, 0.03728728952131943],
               [0.556463206310225, 0.14027595183361663, 0.2731537988657907],
               [0.4093837966398032, 0.8015225687789814, 0.8033567296903834],
               [0.527442563956637, 0.902232617214431, 0.7066626674362227],
               [0.9058355503297827, 0.34983989180213004, 0.8353262183839384],
               [0.7108382186953104, 0.08591307895133471, 0.21434688012521974],
               [0.22757345065207668, 0.7943075496583976, 0.2992305547627421],
               [0.20454109788173636, 0.8251670332103687, 0.012981987094547232],
               [0.7672562637297392, 0.005429019973062554, 0.022163616037108702],
               [0.37487345910117564, 0.5086240194440863, 0.9061216063654387],
               [0.9878004014101087, 0.006345852772772331, 0.17499753379350858],
               [0.030061528704491303, 0.1409704315546606, 0.3337131835834506],
               [0.5022506782611504, 0.5448435505388706, 0.40584238936140726],
               [0.39560774627423445, 0.8905943695833262, 0.5850815030921116],
               [0.058615671926786406, 0.5365713844300387, 0.1620457551256279],
               [0.41843842882069693, 0.1536005983609976, 0.3127878501592438],
               [0.05947621790155899, 0.5412421167331932, 0.2611322146455659],
               [0.5196159938235607, 0.7066461551682705, 0.970261497412556],
               [0.30443031606149007, 0.45158581060034975, 0.4331841153149706],
               [0.8848298403933996, 0.7241791700943656, 0.8917110054596072],
               [0.5720260591898779, 0.3072801598203052, 0.8891066705989902],
               [0.13964015336177327, 0.2531778096760302, 0.5703756837403124],
               [0.2156307542329836, 0.4139947500641685, 0.87051676884144],
               [0.10800455881891169, 0.05554646035458266, 0.2947027428551443],
               [0.35198009410633857, 0.365849666213808, 0.06525787683513773],
               [0.5223264108118847, 0.9032195574351178, 0.28579084943315025],
               [0.7607724246546966, 0.3087194381828555, 0.6253235528354899],
               [0.5060485442077824, 0.19173600467625274, 0.9931175692203702],
               [0.5131805830323746, 0.07719515392040577, 0.923212006754969],
               [0.3629762141280106, 0.02429179642710888, 0.6963754952399983],
               [0.7542592485456767, 0.6478893299494212, 0.3424965345400731],
               [0.49944574453364454, 0.6775665366832825, 0.33758796076989583],
               [0.010621818120767679, 0.8221571611173205, 0.5186257457566332],
               [0.5857910304290109, 0.7178133992025467, 0.9729243483606071],
               [0.16987399482717613, 0.9942570210657463, 0.18120758122552927],
               [0.016362572521240848, 0.17582788603087263, 0.7255176922640298],
               [0.10981764283706419, 0.9078582203470377, 0.7638063718334003],
               [0.9252097840441119, 0.3330197086990039, 0.27888705301420136],
               [0.12769972651171546, 0.11121470804891687, 0.12710743734391716],
               [0.5753520518360334, 0.2763862879599456, 0.6115636613363361]]
YTVIS_CATEGORIES_2021 = {
    1: "airplane",
    2: "bear",
    3: "bird",
    4: "boat",
    5: "car",
    6: "cat",
    7: "cow",
    8: "deer",
    9: "dog",
    10: "duck",
    11: "earless_seal",
    12: "elephant",
    13: "fish",
    14: "flying_disc",
    15: "fox",
    16: "frog",
    17: "giant_panda",
    18: "giraffe",
    19: "horse",
    20: "leopard",
    21: "lizard",
    22: "monkey",
    23: "motorbike",
    24: "mouse",
    25: "parrot",
    26: "person",
    27: "rabbit",
    28: "shark",
    29: "skateboard",
    30: "snake",
    31: "snowboard",
    32: "squirrel",
    33: "surfboard",
    34: "tennis_racket",
    35: "tiger",
    36: "train",
    37: "truck",
    38: "turtle",
    39: "whale",
    40: "zebra",
}
YTVIS_COLOR_2021 = {
    1: [106, 0, 228],
    2: [174, 57, 255],
    3: [255, 109, 65],
    4: [0, 0, 192],
    5: [0, 0, 142],
    6: [255, 77, 255],
    7: [120, 166, 157],
    8: [209, 0, 151],
    9: [0, 226, 252],
    10: [179, 0, 194],
    11: [174, 255, 243],
    12: [110, 76, 0],
    13: [73, 77, 174],
    14: [250, 170, 30],
    15: [0, 125, 92],
    16: [107, 142, 35],
    17: [0, 82, 0],
    18: [72, 0, 118],
    19: [182, 182, 255],
    20: [255, 179, 240],
    21: [119, 11, 32],
    22: [0, 60, 100],
    23: [0, 0, 230],
    24: [130, 114, 135],
    25: [165, 42, 42],
    26: [220, 20, 60],
    27: [100, 170, 30],
    28: [183, 130, 88],
    29: [134, 134, 103],
    30: [5, 121, 0],
    31: [133, 129, 255],
    32: [188, 208, 182],
    33: [145, 148, 174],
    34: [255, 208, 186],
    35: [166, 196, 102],
    36: [0, 80, 100],
    37: [0, 0, 70],
    38: [0, 143, 149],
    39: [0, 228, 0],
    40: [199, 100, 0],
}


def draw_information(im_in1, insid, is_gt=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    margin = 5
    thickness = 1

    im_in = copy.deepcopy(im_in1)
    # BGR for three bad cases
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    if is_gt:
        color = (255, 105, 65)
    else:
        color = (13, 23, 227)

    x = y = 0

    for _nt, text in enumerate(insid):
        size = cv2.getTextSize(text, font, font_scale, thickness)

        text_width = size[0][0]
        text_height = size[0][1]

        im_in[y:text_height + margin + y, :text_width + margin, :] = np.array(
            [220, 220, 220])  # np.zeros((10,10,3))

        x = margin
        if _nt==0:
            y = text_height + y
        else:
            y = text_height + y + margin

        im_in = cv2.putText(np.ascontiguousarray(im_in), text, (x, y), font, font_scale, color, thickness)
    return im_in


def toRLE(mask: object, w: int, h: int):
    """
    Borrowed from Pycocotools:
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """

    if type(mask) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(mask, h, w)
        return mask_utils.merge(rles)
    elif type(mask['counts']) == list:
        # uncompressed RLE
        return mask_utils.frPyObjects(mask, h, w)
    else:
        return mask


# visualizer for vis
class Visualizer:
    def __init__(self, ex, video_id, video_files, image_root, save_root):
        self.save_root = save_root
        self.ex = ex
        self.video_id = video_id
        self.video_files = video_files
        self.image_root = image_root
        if self.image_root != None:
            self._read_images()

    def _read_images(self):
        self.images = []
        for fname in self.video_files:
            self.images.append(cv2.imread(os.path.join(self.image_root, fname)))

        self.width = self.images[0].shape[1]
        self.height = self.images[0].shape[0]

    def _coloring_mask(self, mask, img1, color, alpha):
        '''

        Args:
            mask:
            img:
        Returns: video with masks

        '''
        img = copy.deepcopy(img1)
        color = np.array(color, dtype=np.uint8)
        mask = mask.astype(np.bool)
        img[mask] = img[mask] * (1 - alpha) + alpha * color
        return img

    # draw instance prediction by errortype
    def draw(self, pred, error_type):
        if self.image_root == None:
            pass
        if error_type != 'Miss':
            print('--processing video:', self.video_id, '  prediction/gt:', pred['_id'], '  save type:', error_type)
            # creat folder
            save_path = os.path.join(self.save_root, 'video' + str(self.video_id), YTVIS_CATEGORIES_2021[pred['class']],
                                     error_type, )
        else:
            print('--processing video:', self.video_id, '  prediction:', pred, '  error type:', error_type)
            # creat folder
            save_path = os.path.join(self.save_root, 'video' + str(self.video_id),
                                     YTVIS_CATEGORIES_2021[self.ex.gt[pred]['class']],
                                     error_type, )

        os.makedirs(save_path, exist_ok=True)

        _files = list(os.listdir(save_path))
        if error_type != 'Miss':
            save_path = os.path.join(save_path, str(len(_files)) + '_score-' + str(round(pred['score'], 2)) + '_iou-' +
                                     str(round(pred['iou'], 2)))
            # generate color
            colors = [x for x in YTVIS_COLOR_2021[pred['class']]][::-1]
        else:
            save_path = os.path.join(save_path, str(len(_files)) + '_gtid-' + str(self.ex.gt[pred]['_id']))
            # generate color
            colors = [x for x in YTVIS_COLOR_2021[self.ex.gt[pred]['class']]][::-1]
        os.makedirs(save_path, exist_ok=True)

        alpha = 0.5

        # print('get masks')

        masks = [None, None]  # [gt_mask,dt_mask]
        if error_type == 'Miss':
            masks[0] = self.ex.gt[pred]['mask']
            txt_gt = ['gt_id:' + str(self.ex.gt[pred]['_id']),
                      'label:' + YTVIS_CATEGORIES_2021[self.ex.gt[pred]['class']],
                      'used:' + str(self.ex.gt[pred]['used'])]

        elif len(self.ex.gt) != 0 and pred['vis_gt_idx'] != None:
            masks[0] = self.ex.gt[pred['vis_gt_idx']]['mask']
            txt_gt = ['gt_id:' + str(self.ex.gt[pred['vis_gt_idx']]['_id']),
                      'label:' + YTVIS_CATEGORIES_2021[pred['class']],
                      'used:' + str(self.ex.gt[pred['vis_gt_idx']]['used'])]
        # if pred exists
        if error_type != 'Miss':
            masks[1] = pred['mask']

            txt_pred = ['label:' + YTVIS_CATEGORIES_2021[pred['class']], 'iou:' + str(round(pred['iou'], 2)),
                        'score:' + str(round(pred['score'], 2))]

        # print('drawing')
        # get masks and dets
        if masks[0] != None:
            gtmask = []
            for idx, gtm in enumerate(masks[0]):
                if gtm != None:
                    gtm = mask_utils.decode(toRLE(gtm, self.width, self.height))
                else:
                    gtm = np.zeros((self.height, self.width))
                gtm = self._coloring_mask(gtm, self.images[idx], colors, alpha)
                gtmask.append(draw_information(gtm, txt_gt, True))

            masks[0] = gtmask
        # if pred exists
        if error_type != 'Miss':
            dtmask = []
            for idx, dtm in enumerate(masks[1]):
                dtm = mask_utils.decode(dtm)
                dtm = self._coloring_mask(dtm, self.images[idx], colors, alpha)
                dtmask.append(draw_information(dtm, txt_pred))
            masks[1] = dtmask

        # print('saving')
        # save images
        final_imgs = []

        # gt and perd both exist
        if masks[0] != None and error_type != 'Miss':
            for dtm, gtm in zip(masks[1], masks[0]):
                final_imgs.append(np.concatenate([dtm, (np.ones_like(dtm) * 255)[:10], gtm], axis=0))
        # only pred exists
        elif masks[0] == None and error_type != 'Miss':
            final_imgs = masks[1]
        # only gt exists
        elif masks[0] != None and error_type == 'Miss':
            final_imgs = masks[0]

        for fnum, fim in enumerate(final_imgs):
            cv2.imwrite(os.path.join(save_path, 'frame-' + str(fnum) + '.jpg'), fim)
