import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
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
import kornia

from util.loader.CityLoader import CityLoader
from util.loader.GTA5Loader import GTA5Loader
from util.loader.augmentations import Compose, RandomHorizontallyFlip, RandomSized_and_Crop, RandomCrop, Compose_Pseudo, RandomHorizontallyFlip_Pseudo, RandomSized_and_Crop_Pseudo, RandomCrop_Pseudo
from util.metrics import runningScore
from util.loss import cross_entropy2d, VGGLoss
from model.model import SharedEncoder, ImgDecoder, Discriminator, SegDiscriminator
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models

# Data-related
LOG_DIR = './log_stg2'
GEN_IMG_DIR = './generated_imgs_stg2'

GTA5_DATA_PATH = './data/GTA5'
CITY_DATA_PATH = './data/Cityscapes'
DATA_LIST_PATH_GTA5 = './util/loader/gta5_list/train_modified.txt'
DATA_LIST_PATH_CITY_IMG = './util/loader/cityscapes_list/train.txt'
DATA_LIST_PATH_CITY_LBL = './util/loader/cityscapes_list/train_label.txt'
DATA_LIST_PATH_VAL_IMG  = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL  = './util/loader/cityscapes_list/val_label.txt'

# Hyper-parameters
CUDA_DIVICE_ID = '0'

parser = argparse.ArgumentParser(description='TridentAdapt\
	for domain adaptive semantic segmentation')
parser.add_argument('--dump_logs', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='the path to where you save plots and logs.')
parser.add_argument('--gen_img_dir', type=str, default=GEN_IMG_DIR, help='the path to where you save translated images and segmentation maps.')
parser.add_argument('--gta5_data_path', type=str, default=GTA5_DATA_PATH, help='the path to GTA5 dataset.')
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to Cityscapes dataset.')
parser.add_argument('--data_list_path_gta5', type=str, default=DATA_LIST_PATH_GTA5)
parser.add_argument('--data_list_path_city_img', type=str, default=DATA_LIST_PATH_CITY_IMG)
parser.add_argument('--data_list_path_city_lbl', type=str, default=DATA_LIST_PATH_CITY_LBL)
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)

parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)

args = parser.parse_args()

print ('cuda_device_id:', ','.join(args.cuda_device_id))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
    
if not os.path.exists(args.gen_img_dir):
    os.makedirs(args.gen_img_dir)

if args.dump_logs == True:
	old_output = sys.stdout
	sys.stdout = open(os.path.join(args.log_dir, 'output.txt'), 'w')

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

num_classes = 19
source_input_size = [720, 1280]
target_input_size = [512, 1024]
batch_size = 4

max_epoch = 150
num_steps  = 250000
num_calmIoU = 1000

learning_rate_seg = 2.5e-4
learning_rate_d   = 1e-4
learning_rate_g = 1e-3
learning_rate_dis = 1e-4
power             = 0.9
weight_decay      = 0.0005

lambda_seg = 0.1
lambda_adv_target1 = 0.01#0.0002
lambda_adv_target2 = 0.01#0.001

source_channels = 3
target_channels = 3
bottleneck_channel = 512

# Setup Augmentations
gta5_data_aug = Compose([RandomHorizontallyFlip(),
                         RandomSized_and_Crop([256, 512])
                         ])

city_data_aug = Compose_Pseudo([RandomHorizontallyFlip_Pseudo(),
                         RandomSized_and_Crop_Pseudo([256, 512])
                        ])
# ==== DataLoader ====
gta5_set   = GTA5Loader(args.gta5_data_path, args.data_list_path_gta5, max_iters=num_steps* batch_size, crop_size=source_input_size, transform=gta5_data_aug, mean=IMG_MEAN)
source_loader= torch_data.DataLoader(gta5_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

city_set   = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl, max_iters=num_steps* batch_size, crop_size=target_input_size, transform=city_data_aug, mean=IMG_MEAN, set='train', use_pseudo = True)
target_loader= torch_data.DataLoader(city_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

val_set   = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[512, 1024], mean=IMG_MEAN, set='val')
val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

sourceloader_iter = enumerate(source_loader)
targetloader_iter = enumerate(target_loader)

# Setup Metrics
cty_running_metrics = runningScore(num_classes)

model_dict = {}

# Setup Model
print ('building models ...')
enc_shared = SharedEncoder().cuda()
dclf1      = SegDiscriminator().cuda()
dclf2      = SegDiscriminator().cuda()

g_s      = ImgDecoder(bottleneck_channel).cuda()
g_t      = ImgDecoder(bottleneck_channel).cuda()
dis_s2t    = Discriminator().cuda()
dis_t2s    = Discriminator().cuda()

model_dict['enc_shared'] = enc_shared
model_dict['dclf1'] = dclf1
model_dict['dclf2'] = dclf2

model_dict['g_s'] = g_s
model_dict['g_t'] = g_t
model_dict['dis_s2t'] = dis_s2t
model_dict['dis_t2s'] = dis_t2s


enc_shared_opt = optim.SGD(enc_shared.optim_parameters(learning_rate_seg), lr=learning_rate_seg, momentum=0.9, weight_decay=weight_decay)
dclf1_opt = optim.Adam(dclf1.parameters(), lr=learning_rate_d, betas=(0.9, 0.99))
dclf2_opt = optim.Adam(dclf2.parameters(), lr=learning_rate_d, betas=(0.9, 0.99))

g_s_opt = optim.Adam(g_s.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
g_t_opt = optim.Adam(g_t.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
dis_s2t_opt = optim.Adam(dis_s2t.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999))
dis_t2s_opt = optim.Adam(dis_t2s.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999))

seg_opt_list  = []
dclf_opt_list = []
g_opt_list  = []
dis_opt_list  = []

# Optimizer list for quickly adjusting learning rate
seg_opt_list.append(enc_shared_opt)
dclf_opt_list.append(dclf1_opt)
dclf_opt_list.append(dclf2_opt)

g_opt_list.append(g_s_opt)
g_opt_list.append(g_t_opt)
dis_opt_list.append(dis_s2t_opt)
dis_opt_list.append(dis_t2s_opt)

load_models(model_dict, './weights/')

cudnn.enabled   = True
cudnn.benchmark = False

mse_loss = nn.MSELoss(size_average=True).cuda()
bce_loss = nn.BCEWithLogitsLoss().cuda()
sg_loss  = cross_entropy2d
L1Loss = nn.L1Loss().cuda()
VGG_loss = VGGLoss()


upsample_256 = nn.Upsample(size=[256, 512], mode='bilinear', align_corners=True)
upsample_360 = nn.Upsample(size=[360, 640], mode='bilinear', align_corners=True)
upsample_512 = nn.Upsample(size=[512, 1024], mode='bilinear', align_corners=True)

true_label = 1
fake_label = 0

i_iter_tmp  = []
epoch_tmp = []

loss_recon_s_tmp  = []
loss_recon_t_tmp  = []

prob_dclf1_real1_tmp = []
prob_dclf1_fake1_tmp = []
prob_dclf1_fake2_tmp = []
prob_dclf2_real1_tmp = []
prob_dclf2_fake1_tmp = []
prob_dclf2_fake2_tmp = []

loss_sim_sg_tmp = []

prob_dis_s2t_real1_tmp = []
prob_dis_s2t_fake1_tmp = []
prob_dis_s2t_fake2_tmp = []
prob_dis_t2s_real1_tmp = []
prob_dis_t2s_fake1_tmp = []
prob_dis_t2s_fake2_tmp = []

City_tmp  = [] 

dclf1.train()
dclf2.train()
enc_shared.train()

g_s.train()
g_t.train()
dis_s2t.train()
dis_t2s.train()

best_iou = 0
best_iter= 0
for i_iter in range(num_steps):
    print (i_iter)
    sys.stdout.flush()

    enc_shared.train()
    adjust_learning_rate(seg_opt_list , base_lr=learning_rate_seg, i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(dclf_opt_list, base_lr=learning_rate_d  , i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(g_opt_list , base_lr=learning_rate_g, i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(dis_opt_list , base_lr=learning_rate_dis, i_iter=i_iter, max_iter=num_steps, power=power)

    # ==== sample data ====
    idx_s, source_batch = next(sourceloader_iter)
    idx_t, target_batch = next(targetloader_iter)

    source_data, source_label = source_batch
    target_data, target_label, target_label_pseudo = target_batch

    sdatav = Variable(source_data).cuda()
    slabelv = Variable(source_label).cuda()
    tdatav = Variable(target_data).cuda()
    tlabelv = Variable(target_label)
    tlabelv_pseudo = Variable(target_label_pseudo).cuda()
    
    # forwarding
    shared_feature_s_shallow, s_pred1, s_pred2, shared_feature_s = enc_shared(sdatav)
    shared_feature_t_shallow, t_pred1, t_pred2, shared_feature_t = enc_shared(tdatav)


    if i_iter < 1:
        rec_s   = g_s(shared_feature_s_shallow.detach())
        rec_t   = g_t(shared_feature_t_shallow.detach())
        rec_t2s = g_s(shared_feature_t_shallow.detach())
        rec_s2t = g_t(shared_feature_s_shallow.detach())

        # self-induced cross-domain augmentation
        _, s_pred1_1, s_pred2_2, shared_feature_s_aug = enc_shared(rec_s2t)
        _, t_pred1_1, t_pred2_2, shared_feature_t_aug = enc_shared(rec_t2s)
    else:
        rec_s = g_s(shared_feature_s_shallow)
        rec_t = g_t(shared_feature_t_shallow)
        rec_t2s = g_s(shared_feature_t_shallow)
        rec_s2t = g_t(shared_feature_s_shallow)

        # self-induced cross-domain augmentation
        _, s_pred1_1, s_pred2_2, shared_feature_s_aug = enc_shared(rec_s2t)
        _, t_pred1_1, t_pred2_2, shared_feature_t_aug = enc_shared(rec_t2s)


    for p in dclf1.parameters():
        p.requires_grad = True
    for p in dclf2.parameters():
        p.requires_grad = True
    for p in dis_s2t.parameters():
        p.requires_grad = True
    for p in dis_t2s.parameters():
        p.requires_grad = True

    # train SegDiscriminator
    # ===== dclf1 =====
    prob_dclf1_real1 = dclf1(F.softmax(upsample_256(s_pred1.detach()), dim=1))
    prob_dclf1_fake1 = dclf1(F.softmax(upsample_256(t_pred1.detach()), dim=1))
    loss_d_dclf1 = mse_loss(prob_dclf1_real1, Variable(torch.FloatTensor(prob_dclf1_real1.data.size()).fill_(true_label)).cuda()).cuda() \
                 + mse_loss(prob_dclf1_fake1, Variable(torch.FloatTensor(prob_dclf1_fake1.data.size()).fill_(fake_label)).cuda()).cuda()
    if i_iter%1 == 0:
        dclf1_opt.zero_grad()
        loss_d_dclf1.backward()
        dclf1_opt.step()

    # ===== dclf2 =====
    prob_dclf2_real1 = dclf2(F.softmax(upsample_256(s_pred2.detach()), dim=1))
    prob_dclf2_fake1 = dclf2(F.softmax(upsample_256(t_pred2.detach()), dim=1))
    loss_d_dclf2 = mse_loss(prob_dclf2_real1, Variable(torch.FloatTensor(prob_dclf2_real1.data.size()).fill_(true_label)).cuda()).cuda() \
                 + mse_loss(prob_dclf2_fake1, Variable(torch.FloatTensor(prob_dclf2_fake1.data.size()).fill_(fake_label)).cuda()).cuda()
    if i_iter%1 == 0:
        dclf2_opt.zero_grad()
        loss_d_dclf2.backward()
        dclf2_opt.step()
    
    # train image discriminator -> LSGAN
    # ===== dis_s2t =====
    if i_iter%1 == 0:
        prob_dis_s2t_real1_list = dis_s2t(tdatav)
        prob_dis_s2t_fake1_list = dis_s2t(rec_s2t.detach())
        loss_d_s2t = 0
        for it, (prob_dis_s2t_real1, prob_dis_s2t_fake1) in enumerate(zip(prob_dis_s2t_real1_list, prob_dis_s2t_fake1_list)):
            loss_d_s2t += torch.mean((prob_dis_s2t_real1 - 1) ** 2) + torch.mean((prob_dis_s2t_fake1 - 0) ** 2)
        dis_s2t_opt.zero_grad()
        loss_d_s2t.backward()
        dis_s2t_opt.step()

    # ===== dis_t2s =====
    if i_iter%1 == 0:
        prob_dis_t2s_real1_list = dis_t2s(sdatav)
        prob_dis_t2s_fake1_list = dis_t2s(rec_t2s.detach())
        loss_d_t2s = 0
        for it, (prob_dis_t2s_real1, prob_dis_t2s_fake1) in enumerate(zip(prob_dis_t2s_real1_list, prob_dis_t2s_fake1_list)):
            loss_d_t2s += torch.mean((prob_dis_t2s_real1 - 1) ** 2) + torch.mean((prob_dis_t2s_fake1 - 0) ** 2)
        dis_t2s_opt.zero_grad()
        loss_d_t2s.backward()
        dis_t2s_opt.step()
    
    for p in dclf1.parameters():
        p.requires_grad = False
    for p in dclf2.parameters():
        p.requires_grad = False
    for p in dis_s2t.parameters():
        p.requires_grad = False
    for p in dis_t2s.parameters():
        p.requires_grad = False
        
    # ==== Image self-reconstruction loss  & feature consistency loss ====
    loss_recon_s = L1Loss(rec_s, sdatav)
    loss_recon_t = L1Loss(rec_t, tdatav)
    loss_recon_self = loss_recon_s + loss_recon_t

    loss_vgg_s = VGG_loss(rec_s, sdatav)
    loss_vgg_t = VGG_loss(rec_t, tdatav)
    loss_vgg_s2t = VGG_loss(rec_s2t, sdatav)
    loss_vgg_t2s = VGG_loss(rec_t2s, tdatav)
    loss_vgg = loss_vgg_s + loss_vgg_t+ 0.5*loss_vgg_s2t + 0.5*loss_vgg_t2s

    loss_grad_s = L1Loss(kornia.sobel(rec_s), kornia.sobel(sdatav))
    loss_grad_t = L1Loss(kornia.sobel(rec_t), kornia.sobel(tdatav))
    loss_grad = loss_grad_s + loss_grad_t


    loss_feature_s = L1Loss(shared_feature_s_aug, shared_feature_s.detach())
    loss_feature_t = L1Loss(shared_feature_t_aug, shared_feature_t.detach())
    loss_recon_feature = loss_feature_s + loss_feature_t


    # ==== SegDiscriminator loss, update Encoder ====
    prob_dclf1_fake2 = dclf1(F.softmax(upsample_256(t_pred1), dim=1))
    dloss_output_seg1 = mse_loss(prob_dclf1_fake2, Variable(torch.FloatTensor(prob_dclf1_fake2.data.size()).fill_(true_label)).cuda())

    prob_dclf2_fake2 = dclf2(F.softmax(upsample_256(t_pred2), dim=1))
    dloss_output_seg2 = mse_loss(prob_dclf2_fake2, Variable(torch.FloatTensor(prob_dclf2_fake2.data.size()).fill_(true_label)).cuda())

    dloss_output_seg = lambda_adv_target1* dloss_output_seg1 + lambda_adv_target2* dloss_output_seg2
    
    # ==== image translation loss====
    # prob_dis_s2t_real2_list = dis_s2t(tdatav)
    prob_dis_s2t_fake2_list = dis_s2t(rec_s2t)
    loss_gen_s2t = 0
    for it, (prob_dis_s2t_fake2) in enumerate(prob_dis_s2t_fake2_list):
        loss_gen_s2t += torch.mean((prob_dis_s2t_fake2 - 1) ** 2)

    # prob_dis_t2s_real2_list = dis_t2s(sdatav)
    prob_dis_t2s_fake2_list = dis_t2s(rec_t2s)
    loss_gen_t2s = 0
    for it, (prob_dis_t2s_fake2) in enumerate(prob_dis_t2s_fake2_list):
        loss_gen_t2s += torch.mean((prob_dis_t2s_fake2 - 1) ** 2)
    loss_image_translation = loss_gen_s2t + loss_gen_t2s
    
    # ==== segmentation loss ====    
    s_pred1 = upsample_256(s_pred1)
    s_pred2 = upsample_256(s_pred2)
    loss_s_sg1 = sg_loss(s_pred1, slabelv)
    loss_s_sg2 = sg_loss(s_pred2, slabelv)
    
    loss_sim_sg = lambda_seg* loss_s_sg1 + loss_s_sg2

    # ==== segmentation loss pseudo target====
    t_pred1_up = upsample_256(t_pred1)
    t_pred2_up = upsample_256(t_pred2)

    loss_t_sg1 = sg_loss(t_pred1_up, tlabelv_pseudo)
    loss_t_sg2 = sg_loss(t_pred2_up, tlabelv_pseudo)

    loss_sim_sg += 0.75*(lambda_seg * loss_t_sg1 + loss_t_sg2)

    
    # ==== tranalated segmentation====
    if i_iter >= 1:
        # check if we have to detach the rec_s2t images
        _, s2t_pred1, s2t_pred2, _ = enc_shared(rec_s2t.detach())
        s2t_pred1 = upsample_256(s2t_pred1)
        s2t_pred2 = upsample_256(s2t_pred2)
        loss_s2t_sg1 = sg_loss(s2t_pred1, slabelv)
        loss_s2t_sg2 = sg_loss(s2t_pred2, slabelv)
        loss_sim_sg += lambda_seg* loss_s2t_sg1 + loss_s2t_sg2

        # ==== segmentation loss pseudo t2s====
        _, t2s_pred1, t2s_pred2, _ = enc_shared(rec_t2s.detach())
        t2s_pred1_up = upsample_256(t2s_pred1)
        t2s_pred2_up = upsample_256(t2s_pred2)

        loss_t2s_sg1 = sg_loss(t2s_pred1_up, tlabelv_pseudo)
        loss_t2s_sg2 = sg_loss(t2s_pred2_up, tlabelv_pseudo)

        loss_sim_sg += 0.75 * (lambda_seg * loss_t2s_sg1 + loss_t2s_sg2)


    # visualize segmentation map
    t_pred2 = upsample_256(t_pred2)

    pred_s = F.softmax(s_pred2, dim=1).data.max(1)[1].cpu().numpy()
    pred_t = F.softmax(t_pred2, dim=1).data.max(1)[1].cpu().numpy()

    map_s  = gta5_set.decode_segmap(pred_s)
    map_t  = city_set.decode_segmap(pred_t)

    gt_s = slabelv.data.cpu().numpy()
    gt_t = tlabelv.data.cpu().numpy()
    gt_t_pseudo = tlabelv_pseudo.data.cpu().numpy()
    gt_s  = gta5_set.decode_segmap(gt_s)
    gt_t  = city_set.decode_segmap(gt_t)
    gt_t_pseudo = city_set.decode_segmap(gt_t_pseudo)

    total_loss = \
              1.0 * loss_sim_sg \
            + 1.0 * dloss_output_seg \
            + 1.0 * loss_recon_self \
            + 0.1 * loss_image_translation \
            + 0.5 * loss_vgg \
            + 0.5 * loss_grad
    if i_iter >= 1:
            total_loss +=  0.1 * loss_recon_feature + 0.2 * loss_sim_sg

    enc_shared_opt.zero_grad()
    g_s_opt.zero_grad()
    g_t_opt.zero_grad()

    total_loss.backward()

    enc_shared_opt.step()
    g_s_opt.step()
    g_t_opt.step()
        
    if i_iter % 25 == 0: 
        i_iter_tmp.append(i_iter)
        print ('Best Iter : '+str(best_iter))
        print ('Best mIoU : '+str(best_iou))
        
        plt.title('prob_s2t')
        prob_dis_s2t_real1_tmp.append(prob_dis_s2t_real1.data[0].cpu().mean())
        prob_dis_s2t_fake1_tmp.append(prob_dis_s2t_fake1.data[0].cpu().mean())
        prob_dis_s2t_fake2_tmp.append(prob_dis_s2t_fake2.data[0].cpu().mean())
        plt.plot(i_iter_tmp, prob_dis_s2t_real1_tmp, label='prob_dis_s2t_real1')
        plt.plot(i_iter_tmp, prob_dis_s2t_fake1_tmp, label='prob_dis_s2t_fake1')
        plt.plot(i_iter_tmp, prob_dis_s2t_fake2_tmp, label='prob_dis_s2t_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_s2t.png'))
        plt.close()

        plt.title('prob_t2s')
        prob_dis_t2s_real1_tmp.append(prob_dis_t2s_real1.data[0].cpu().mean())
        prob_dis_t2s_fake1_tmp.append(prob_dis_t2s_fake1.data[0].cpu().mean())
        prob_dis_t2s_fake2_tmp.append(prob_dis_t2s_fake2.data[0].cpu().mean())
        plt.plot(i_iter_tmp, prob_dis_t2s_real1_tmp, label='prob_dis_t2s_real1')
        plt.plot(i_iter_tmp, prob_dis_t2s_fake1_tmp, label='prob_dis_t2s_fake1')
        plt.plot(i_iter_tmp, prob_dis_t2s_fake2_tmp, label='prob_dis_t2s_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_t2s.png'))
        plt.close()
        
        plt.title('recon self loss')
        loss_recon_s_tmp.append(loss_recon_s.item())
        loss_recon_t_tmp.append(loss_recon_t.item())
        plt.plot(i_iter_tmp, loss_recon_s_tmp, label='loss_recon_s')
        plt.plot(i_iter_tmp, loss_recon_t_tmp, label='loss_recon_t')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'recon_loss.png'))
        plt.close()

        
        plt.title('prob_dclf1')
        prob_dclf1_real1_tmp.append(prob_dclf1_real1.data[0].cpu().mean())
        prob_dclf1_fake1_tmp.append(prob_dclf1_fake1.data[0].cpu().mean())
        prob_dclf1_fake2_tmp.append(prob_dclf1_fake2.data[0].cpu().mean())
        plt.plot(i_iter_tmp, prob_dclf1_real1_tmp, label='prob_dclf1_real1')
        plt.plot(i_iter_tmp, prob_dclf1_fake1_tmp, label='prob_dclf1_fake1')
        plt.plot(i_iter_tmp, prob_dclf1_fake2_tmp, label='prob_dclf1_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_dclf1.png'))
        plt.close()

        plt.title('prob_dclf2')
        prob_dclf2_real1_tmp.append(prob_dclf2_real1.data[0].cpu().mean())
        prob_dclf2_fake1_tmp.append(prob_dclf2_fake1.data[0].cpu().mean())
        prob_dclf2_fake2_tmp.append(prob_dclf2_fake2.data[0].cpu().mean())
        plt.plot(i_iter_tmp, prob_dclf2_real1_tmp, label='prob_dclf2_real1')
        plt.plot(i_iter_tmp, prob_dclf2_fake1_tmp, label='prob_dclf2_fake1')
        plt.plot(i_iter_tmp, prob_dclf2_fake2_tmp, label='prob_dclf2_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_dclf2.png'))
        plt.close()

        plt.title('segmentation_loss')
        loss_sim_sg_tmp.append(loss_sim_sg.item())
        plt.plot(i_iter_tmp, loss_sim_sg_tmp, label='loss_sim_sg')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'segmentation_loss.png'))
        plt.close()

        plt.title('mIoU')
        plt.plot(epoch_tmp, City_tmp, label='City')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'mIoU.png'))
        plt.close()
        
    if i_iter%500 == 0 :
        imgs_s = torch.cat(((sdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_s[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_s2t[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((map_s.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((gt_s.transpose((0, 3, 1, 2)))))), 0)
        imgs_s = vutils.make_grid(imgs_s.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_s = np.clip(imgs_s*255,0,255).astype(np.uint8)
        imgs_s = imgs_s.transpose(1,2,0)
        imgs_s = Image.fromarray(imgs_s)
        filename = '%05d_source.jpg' % i_iter
        imgs_s.save(os.path.join(args.gen_img_dir, filename))
        
        imgs_t = torch.cat(((tdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_t[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_t2s[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((map_t.transpose((0, 3, 1, 2))))),Variable(torch.Tensor((gt_t_pseudo.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((gt_t.transpose((0, 3, 1, 2)))))), 0)
        imgs_t = vutils.make_grid(imgs_t.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_t = np.clip(imgs_t*255,0,255).astype(np.uint8)
        imgs_t = imgs_t.transpose(1,2,0)
        imgs_t = Image.fromarray(imgs_t)
        filename = '%05d_target.jpg' % i_iter
        imgs_t.save(os.path.join(args.gen_img_dir, filename))

    if i_iter % num_calmIoU == 0:
        enc_shared.eval()
        print ('evaluating models ...')
        for i_val, (images_val, labels_val) in tqdm(enumerate(val_loader)):
            images_val = Variable(images_val.cuda(), requires_grad=False)
            labels_val = Variable(labels_val, requires_grad=False)

            _, _, pred, _ = enc_shared(images_val)
            pred = upsample_512(pred)
            pred = pred.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            cty_running_metrics.update(gt, pred)
            
        cty_score, cty_class_iou = cty_running_metrics.get_scores()
        
        for k, v in cty_score.items():
            print(k, v)
            
        cty_running_metrics.reset()
        City_tmp.append(cty_score['Mean IoU : \t'])
        epoch_tmp.append(i_iter)
        if i_iter % 10000 == 0 and i_iter != 0:
        	save_models(model_dict, './weights_' + str(i_iter))

        if cty_score['Mean IoU : \t'] > best_iou:
            best_iter = i_iter
            best_iou = cty_score['Mean IoU : \t']
            save_models(model_dict, './weights/')
