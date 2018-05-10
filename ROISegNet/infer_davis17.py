import os,sys
sys.path.append("../caffe")
sys.path.append("../caffe/python")
sys.path.append("../caffe/python/caffe")

sys.path.insert(0, "../../fcn_python/")
sys.path.insert(0, "../../python_layers/")

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
        tmp = feat2 - np.array(feat1[i,...])
        tmp = tmp**2
        tmp = np.sum(tmp, 1)
        dist[i,...] = sqrt(tmp)

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


feat_dir = 'featpool_davis17/'
det_dir  = '../siamese-fc-master/tracking/DAVIS2017/'

davis_dir = '../data/DAVIS2017/'
split_f   = '{}/ImageSets/2017/val.txt'.format(davis_dir)

file_out  = 'results/'

indices = open(split_f, 'r').read().splitlines()
print >> sys.stderr, 'Total Number of Images: {}'.format(len(indices))


model           = sys.argv[1]
deploy_proto    = sys.argv[2]
file_out        = sys.argv[3]
selection       = np.int(sys.argv[4])
device_id       = np.int(sys.argv[5])



caffe.set_device(device_id)
caffe.set_mode_gpu()
net = caffe.Net(deploy_proto, model, caffe.TEST)

for idx in range(len(indices)):
 clip = indices[idx].split(' ')[0].split('/')[-2]
 ann_img = '{}/Annotations/2017/{}/00000.png'.format(davis_dir, clip)
 ann_img = Image.open(ann_img)
 ann_img = np.array(ann_img)
 num_obj = np.int(np.max(ann_img))

 if os.path.exists(file_out) == False:
     os.mkdir(file_out)

 if os.path.exists('{}/{}'.format(file_out, clip)) == False:
    os.mkdir('{}/{}'.format(file_out, clip))

 im_name  = '{}/{}'.format(davis_dir, indices[idx].split(' ')[0])
 ann_name = '{}/{}'.format(davis_dir, indices[idx].split(' ')[1])

 img_name = indices[idx].split(' ')[1]
 ss = img_name.split('/')
 ss = ss[len(ss)-1]
 ss = ss[0:len(ss)-4]


 for obj_id  in range(num_obj):	
    feat_name   = '{}/{}_{}.mat'.format(feat_dir,clip,obj_id+1)
    data        = io.loadmat(feat_name)
    feat_pool   = data['feats']
	 
    det_name     = '{}/{}_{}.mat.mat'.format(det_dir, clip, obj_id + 1)
    if os.path.exists(det_name) == False:
       continue
    img_idx = np.int(ss)

    feat_name    = '{}/{}/{}_{}.mat'.format(file_out, clip, ss, obj_id+1)
    if os.path.exists(feat_name) :
       continue
    data = io.loadmat(det_name)

    dets = data['boxes']
    dets = dets[0:dets.shape[0],img_idx,0:dets.shape[2]]
    dets = np.round(dets)

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

    NUM = 150
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
       

        sim = func_dist(feat_pool, outs1)
        io.savemat(feat_name, {'mask': outs2,'sim': sim, 'rois': input_rois})  
        
