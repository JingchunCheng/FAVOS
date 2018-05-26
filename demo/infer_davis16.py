import os,sys
sys.path.append("../caffe")
sys.path.append("../caffe/python")
sys.path.append("../caffe/python/caffe")

import caffe

import numpy as np
from PIL import Image
import scipy.io

from scipy.misc import imresize

import os
from scipy import io

import shutil

from numpy import *
import math
import random



def load_image(im_name):
    im = Image.open(im_name)
    print >> sys.stderr, 'loading {}'.format(im_name)
    return im


def func_dist(feat1, feat2):
    dist = np.zeros([feat1.shape[0], feat2.shape[0]], dtype=np.float32)
    for i in range(feat1.shape[0]):
        for j in range(feat2.shape[0]):
            vector1 = np.float32(mat(feat1[i,...]))
            vector2 = np.float32(mat(feat2[j,...]))
            
            dist[i][j] = sqrt((vector1-vector2)*((vector1-vector2).T))

    return dist


def get_img_rois(img, rois, pool_h, pool_w):
    box_num = rois.shape[0]
    img_rois = np.zeros((box_num, 3, pool_h, pool_w))
    for i in range(box_num):
            bb   = rois[i,1:5]
            ROI  = img.crop(bb)
            ROI  = ROI.resize((np.int(pool_w), np.int(pool_h)))
            in_  = np.array(ROI, dtype=np.float32)
            in_  = in_[:,:,::-1]
            in_ -= np.array((104.00698793,116.66876762,122.67891434))
            ROI  = in_.transpose((2,0,1))
            img_rois[i,...] = ROI

    return img_rois



def get_bbox(label):
    label = np.array(label, dtype=np.uint8)
    pos   = np.where(label >0)
    if len(pos[0])<100:
       print >> sys.stderr, 'escape very small object {}'.format(obj_id)
       bb = [0, 0, 0, 0]
       return bb
    else:
       bb     = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
       print>>sys.stderr, 'gt = {}; area = {};'.format(bb, len(pos[0]))
    return bb


def load_label(im_name):
    print >> sys.stderr, 'loading {}'.format(im_name)
    im = Image.open(im_name)
    label = np.array(im, dtype=np.uint8)
    return label


davis_dir = '../data/DAVIS2016/'
davis_dir = '/home/chengjc/DeepParticle-master/data/DAVIS2016/'
file_out  = '../results/res_part/'

model           = sys.argv[1]
deploy_proto    = sys.argv[2]
cls_name        = sys.argv[3]
device_id       = np.int(sys.argv[4])

caffe.set_device(device_id)
caffe.set_mode_gpu()
net = caffe.Net(deploy_proto, model, caffe.TEST)

img_path = '{}/JPEGImages/480p/{}'.format(davis_dir, cls_name)
images    = os.listdir(img_path)
images    = sorted(images)

det_name     = '{}/{}.mat.mat'.format(det_dir, cls_name)
data         = io.loadmat(det_name)
Dboxes       = data['boxes']


for idx in range(len(images)):

    if os.path.exists(file_out) == False:
        os.mkdir(file_out)

    if os.path.exists('{}/{}'.format(file_out, cls_name)) == False:
        os.mkdir('{}/{}'.format(file_out, cls_name))

    im_name  = '{}/JPEGImages/480p/{}/{}'.format(davis_dir, cls_name, images[idx])

    ss = images[idx].split('.jpg')
    ss = ss[0]
 
    img_idx = np.int(ss) 
    dets    = Dboxes[0:Dboxes.shape[0],img_idx,0:Dboxes.shape[2]]
    dets    = np.round(dets)

    pos  = np.where(dets<0)

    for k in range(len(pos[0])):
        dets[pos[0][k], pos[1][k]] = 0

    input_roi = np.zeros((dets.shape[0],5))
    for k in range(dets.shape[0]):
        input_roi[k, 1] = dets[k, 0]
        input_roi[k, 2] = dets[k, 1]
        input_roi[k, 3] = dets[k, 2]
        input_roi[k, 4] = dets[k, 3]

    img1 = load_image(im_name)
    pool_w  = 80
    pool_h  = 80

    NUM = 200
    if  dets.shape[0] > 0: 
        input_rois = input_roi
        print >> sys.stderr,'box num = {}'.format(dets.shape[0])
        outs2  = np.zeros([dets.shape[0],2,pool_w,pool_h], dtype = np.float32)
        for k in range( np.int(np.ceil( dets.shape[0]*1.0/NUM)) ):
            input_roi = input_rois[k*NUM:np.min([(k+1)*NUM,dets.shape[0]]), ...]
            datas     = get_img_rois(img1, input_roi, pool_h, pool_w)
            net.blobs['datas'].reshape(*datas.shape)
            net.blobs['datas'].data[...]  = datas
            net.forward()
            out1  = net.blobs['feat'].data
            out2  = net.blobs['pred_mask'].data

            if k == 0:
               feat_len = out1.shape[1]
               outs1  = np.zeros([dets.shape[0], feat_len], dtype = np.float32)
               print >> sys.stderr, 'Feature length = {}, Candidate number = {}'.format(feat_len, dets.shape[0])

            outs1[k*NUM:np.min([(k+1)*NUM,dets.shape[0]]),...]  = out1.reshape([out1.shape[0],out1.shape[1]])
            outs2[k*NUM:np.min([(k+1)*NUM,dets.shape[0]]),...]  = out2
            print>>sys.stderr, 'Processing {}->{}'.format(k*NUM,  np.min([(k+1)*NUM,dets.shape[0]]))
       
        io.savemat('{}/{}/{}.mat'.format(file_out, cls_name, ss), {'mask': outs2, 'rois': input_rois})  
        
