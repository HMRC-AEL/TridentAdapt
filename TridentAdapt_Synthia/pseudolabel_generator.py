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

from model.model import SharedEncoder,ImgDecoder
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models
from util.loader.CityLoader import CityLoader


num_classes = 16
CITY_DATA_PATH = './data/Cityscapes'
DATA_LIST_PATH_TRAIN_IMG = './util/loader/cityscapes_list/train.txt'
DATA_LIST_PATH_TRAIN_LBL  = './util/loader/cityscapes_list/train_label.txt'
WEIGHT_DIR = './weights'
OUTPUT_DIR = './pseudo_train'
DEFAULT_GPU = 0
bottleneck_channel = 512
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='TridentAdapt\
	for domain adaptive semantic segmentation')
parser.add_argument('weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to cityscapes.')
parser.add_argument('--data_list_path_train_img', type=str, default=DATA_LIST_PATH_TRAIN_IMG)
parser.add_argument('--gpu', type=str, default=DEFAULT_GPU)
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
parser.add_argument('--data_list_path_train_lbl', type=str, default=DATA_LIST_PATH_TRAIN_LBL)


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 60, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


args = parser.parse_args()

train_set   = CityLoader(args.city_data_path, args.data_list_path_train_img, args.data_list_path_train_lbl, max_iters=None, crop_size=[512, 1024], mean=IMG_MEAN, set='train', return_name = True)
train_loader= torch_data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


#upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear', align_corners=True)
#upsample_512 = nn.Upsample(size=[512, 1024], mode='bilinear', align_corners=True)

model_dict = {}

enc_shared = SharedEncoder().cuda()
model_dict['enc_shared'] = enc_shared

load_models(model_dict, args.weight_dir)

enc_shared.eval()

predicted_label = np.zeros((len(train_loader), 512, 1024))
predicted_prob = np.zeros((len(train_loader), 512, 1024))
image_name = []

for index, batch in enumerate(train_loader):
    if index % 100 == 0:
        print('%d processd' % index)
    image, _, name = batch
    shared_feature_t_shallow, _, output, _ = enc_shared(Variable(image).cuda())
    output = nn.functional.softmax(output, dim=1)
    output = nn.functional.interpolate(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)

    label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
    predicted_label[index] = label.copy()
    predicted_prob[index] = prob.copy()
    image_name.append(name[0])

thres = []
for i in range(16):
    x = predicted_prob[predicted_label == i]
    if len(x) == 0:
        thres.append(0)
        continue
    x = np.sort(x)
    thres.append(x[np.int(np.round(len(x) * 0.5))])
print(thres)
thres = np.array(thres)
thres[thres > 0.9] = 0.9
print(thres)
for index in range(len(train_loader)):
    name = image_name[index]
    label = predicted_label[index]
    prob = predicted_prob[index]
    for i in range(16):
        label[(prob < thres[i]) * (label == i)] = 255
    output = np.asarray(label, dtype=np.uint8)

    output = Image.fromarray(output)
    name = name.split('/')[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output.save(os.path.join(args.output_dir, name))
    '''
    output = colorize_mask(output)

    name = name.split('/')[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output.save(os.path.join(args.output_dir, name))
    '''
