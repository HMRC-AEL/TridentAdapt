import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from collections import Counter
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import torchvision.models as models
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import os

# from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from util.metrics import runningScore
from model.model import SharedEncoder,ImgDecoder
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models
from util.loader.CityLoader import CityLoader
from util.loader.CityTestLoader import CityTestLoader

num_classes = 19
CITY_DATA_PATH = './data/Cityscapes'
DATA_LIST_PATH_VAL_IMG = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL  = './util/loader/cityscapes_list/val_label.txt'
WEIGHT_DIR = './weights'
OUTPUT_DIR = './result_val'
DEFAULT_GPU = 0
bottleneck_channel = 512
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='TridentAdapt \
	for domain adaptive semantic segmentation')
parser.add_argument('weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to cityscapes.')
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--gpu', type=str, default=DEFAULT_GPU)
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


args = parser.parse_args()

val_set   = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[512, 1024], mean=IMG_MEAN, set='val')
val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

val_set0   = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[1024, 2048], mean=IMG_MEAN, set='val')
val_loader0 = torch_data.DataLoader(val_set0, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear', align_corners=True)
upsample_512 = nn.Upsample(size=[512, 1024], mode='bilinear', align_corners=True)

model_dict = {}

enc_shared = SharedEncoder().cuda()
model_dict['enc_shared'] = enc_shared


load_models(model_dict, args.weight_dir)

enc_shared.eval()


cty_running_metrics = runningScore(num_classes)
print('evaluating models ...')
for i_val, ((images_val, labels_val), (images_val0, labels_val0)) in enumerate(zip(val_loader, val_loader0)):
    print(i_val)
    images_val = Variable(images_val.cuda(), requires_grad=False)
    labels_val0 = Variable(labels_val0, requires_grad=False)

    shared_feature_t_shallow, _, pred, _ = enc_shared(images_val)
    pred = upsample_1024(pred)
    pred = pred.data.max(1)[1].cpu().numpy()
    gt = labels_val0.data.cpu().numpy()
    cty_running_metrics.update(gt, pred)
cty_score, cty_class_iou = cty_running_metrics.get_scores()

for k, v in cty_score.items():
    print(k, v)


'''
# uncommenting this part will generate color masks for segmentation
print('generating segmentation images...')
for i_val, (images_val, name) in tqdm(enumerate(val_loader0)):
    images_val = Variable(images_val.cuda(), requires_grad=False)

    _, _, pred, _ = enc_shared(images_val)
    pred = upsample_512(pred)

    pred = pred.data.cpu().numpy()[0]
    pred = pred.transpose(1,2,0)
    pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
    pred = colorize_mask(pred)
    
    name = name[0][0].split('/')[-1]
    if not os.path.exists(args.output_dir):
    	os.makedirs(args.output_dir)
    pred.save(os.path.join(args.output_dir, name))

'''