import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle

from model.deeplab import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet


#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 6
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots/default/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1
LAMBDA_FM = 0.1

PARTIAL_DATA=0.5

SEMI_START=2000
LAMBDA_SEMI=0.1
MASK_T=0.2


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="lambda_fm for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)

    return D_label

criterion = nn.BCELoss()

def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)

    # load pretrained parameters
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print (name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            print('copy {}'.format(name))
    model.load_state_dict(new_params)


    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D.train()
    model_D.cuda(args.gpu)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    if args.partial_data is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    else:
        #sample partial data
        partial_size = int(args.partial_data * train_dataset_size)
        
        if args.partial_id is not None:
            train_ids = pickle.load(open(args.partial_id))
            print('loading train ids from {}'.format(args.partial_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)
        
        pickle.dump(train_ids, open(osp.join(args.snapshot_dir, 'train_id.pkl'), 'wb'))
        
        train_sampler_all = data.sampler.SubsetRandomSampler(train_ids)
        #train_gt_sampler_all = data.sampler.SubsetRandomSampler(train_ids)
        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        #train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader_all = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler_all, num_workers=3, pin_memory=True)
        #trainloader_gt_all = data.DataLoader(train_gt_dataset,
                        #batch_size=args.batch_size, sampler=train_gt_sampler_all, num_workers=16, pin_memory=True)
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler, num_workers=3, pin_memory=True)
        #trainloader_remain = data.DataLoader(train_dataset,
                        #batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=16, pin_memory=True)
        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=3, pin_memory=True)

        #trainloader_remain_iter = iter(trainloader_remain)


    trainloader_all_iter = iter(trainloader_all)
    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)


    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')


    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    y_real_, y_fake_ = Variable(torch.ones(args.batch_size, 1).cuda()), Variable(torch.zeros(args.batch_size, 1).cuda())


    for i_iter in range(20001, args.num_steps):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_fm_value = 0
        loss_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        #for sub_i in range(args.iter_size):

        if i_iter != 0:
            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source
            
            #batch_l = next(trainloader_iter)

            try:
                batch_l = next(trainloader_iter)
            except:
                trainloader_iter = iter(trainloader)
                batch_l = next(trainloader_iter)           
 
            images, labels, _, _ = batch_l
            images = Variable(images).cuda(args.gpu)
            pred_l = interp(model(images))

            loss_seg = loss_calc(pred_l, labels, args.gpu)
           
            if i_iter <= 500:
                loss_seg.backward()
            loss_seg_value += loss_seg.data.cpu().numpy()[0]/args.iter_size

        if i_iter > 500: 
            optimizer.param_groups[0]['lr'] = 2.5e-5
            optimizer_D.param_groups[0]['lr'] = 1e-5
            
            #fm loss calc
            #batch_all = next(trainloader_all_iter)
            
            try:
                batch_all = next(trainloader_all_iter)
            except:
                trainloader_all_iter = iter(trainloader_all)
                batch_all = next(trainloader_all_iter)
            
            images, _, _, _ = batch_all
            images = Variable(images).cuda(args.gpu)
            pred = interp(model(images))
            
            #output of modelD for predictions
            _, D_out_y_pred = model_D(F.softmax(pred))
            
            #output of modelD for ground truth
            #batch_gt = next(trainloader_gt_iter)
            try:
                batch_gt = next(trainloader_gt_iter)
            except:
                trainloader_gt_iter = iter(trainloader_gt)
                batch_gt = next(trainloader_gt_iter) 
            
            _, labels_gt, _, _ = batch_gt
            D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
            _ , D_out_y_gt = model_D(D_gt_v)
             
            fm_loss = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
            
            loss = loss_seg + fm_loss

            # proper normalization
            loss.backward()
            loss_fm_value+= fm_loss.data.cpu().numpy()[0]/args.iter_size
            loss_value += loss.data.cpu().numpy()[0]/args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with pred
            pred = pred.detach()
            D_out_z, _ = model_D(F.softmax(pred))
            y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
            loss_D_fake = criterion(D_out_z, y_fake_) 

            D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
            D_out_z_gt, _ = model_D(D_gt_v)
            y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda()) 
            loss_D_real = criterion(D_out_z_gt, y_real_)
            
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()[0]

        optimizer.step()
        optimizer_D.step()

        print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_D = {3:.3f}, fm_loss = {4:.3f}, loss_g = {5:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_D_value, loss_fm_value, loss_value))
        #print ('fm_loss: ', loss_fm_value, ' g_loss: ', loss_value)

        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()
