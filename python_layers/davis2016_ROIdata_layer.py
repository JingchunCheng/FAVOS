import caffe

import numpy as np
from PIL import Image
from scipy.misc import imresize

import sys

import random

from numpy import *
import math
from scipy.misc import imresize



class Davis2016ROIDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.davis_dir = params['davis_dir']
        self.split     = params['split']
        self.mean      = np.array(params['mean'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        self.fg_rate   = params.get('fg_rate', 0.5)
        self.augment   = params.get('with_augmentation', True)
        self.fg_random = params.get('fg_random', False)
        self.aug_params= np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 480
        self.W         = 854
        self.pool_w    = params.get('pool_w', 7)
        self.pool_h    = params.get('pool_h', 7)
        self.box_num   = params.get('box_num', 100)
        self.fg_thre   = params.get('fg_thre', 0.7)

        if self.augment:
           self.aug_num         = np.int(self.aug_params[0])
           self.max_scale       = self.aug_params[1]
           self.max_rotate      = self.aug_params[2]
           self.max_transW      = self.aug_params[3]
           self.max_transH      = self.aug_params[4]
           self.flip            = (self.aug_params[5]>0)

        # tops
        if len(top) != 3:
            raise Exception("Need to define five tops: data, labels, weights")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/480p/{}.txt'.format(self.davis_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx1 = -1 # we pick idx in reshape
        self.idx2 = -1

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        while True:
            # pick next input
            if self.random:
                self.idx1 = random.randint(0, len(self.indices)-1)
            else:
                self.idx1 += 1
                if self.idx1 == len(self.indices):
                    self.idx1 = 0

            idx1 = self.idx1

            #get clip name
            clip1 = self.indices[idx1].split(' ')[0].split('/')[-2]
            
            if self.augment == False or random.randint(0, self.aug_num) == 0:
               self.img1    = self.load_image(self.indices[idx1].split(' ')[0])
               self.label1  = self.load_label(self.indices[idx1].split(' ')[1])
               self.img1    = self.img1.resize((self.H, self.W))
               self.label1  = imresize(self.label1,  size=(self.H, self.W), interp="nearest")
            else:
               scale       =  (random.random()*2-1) * self.max_scale
               rotation    =  (random.random()*2-1) * self.max_rotate
               trans_w     =  np.int( (random.random()*2-1) * self.max_transW * self.W )
               trans_h     =  np.int( (random.random()*2-1) * self.max_transH * self.H )
               if self.flip:
                  flip     = (random.randint(0,1) > 0)
               else:
                  flip     = False
               self.img1    = self.load_image_transform(self.indices[idx1].split(' ')[0], scale, rotation, trans_h, trans_w, flip)
               self.label1  = self.load_label_transform(self.indices[idx1].split(' ')[1], scale, rotation, trans_h, trans_w, flip)


            if self.scale != 1:
               self.img1   = self.img1.resize((np.int(self.H*self.scale), np.int(self.W*self.scale)))
            print >> sys.stderr, 'SCALE {}'.format(self.scale)

            if np.max(self.label1) == 0:
               continue
           
            if np.sum(self.get_bbox(self.label1)) == 0: 
            # if objects in either image are too small
               continue

            if self.fg_random:
                self.rois  = self.get_rois(self.label1, self.box_num, 1)
            else:
                self.rois  = self.get_bboxes(self.label1, self.box_num)

            self.img_rois  = self.get_img_rois(self.img1,   self.rois)
            self.lab_rois  = self.get_lab_rois(self.label1, self.rois) 
            self.weights   = self.calculate_weight_rois(self.lab_rois)

            break

        # reshape tops
        top[0].reshape(self.box_num, 3, self.pool_h, self.pool_w)
        top[1].reshape(self.box_num, self.pool_h, self.pool_w)
        top[2].reshape(self.box_num, self.pool_h, self.pool_w)



    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.img_rois
        top[1].data[...] = self.lab_rois
        top[2].data[...] = self.weights
 
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.davis_dir, idx))
        im  = im.resize((self.W, self.H))
#        in_ = np.array(im, dtype=np.float32)
#        in_ = in_[:,:,::-1]
#        in_ -= self.mean
#        in_ = in_.transpose((2,0,1))
        return im


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.davis_dir, idx))
        im  = im.resize((self.W, self.H))
        if self.scale != 1:
           im  = im.resize((np.int(self.W*self.scale), np.int(self.H*self.scale)))

        label = np.array(im, dtype=np.uint8)
#        label = label[np.newaxis, ...]
        print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        return label



    def load_image_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
       img_W = np.int( self.W*(1.0 + scale) )
       img_H = np.int( self.H*(1.0 + scale) ) 

       print >> sys.stderr, 'loading {}'.format(idx)
       print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

       im = Image.open('{}/{}'.format(self.davis_dir, idx))
       im    = im.resize((img_W,img_H))
       im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
       im    = im.rotate(rotation)
       if flip:
          im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
       if scale>0:
          box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
          im  = im.crop(box)
       else:
          im  = im.resize((self.W, self.H))
       
#       in_ = np.array(im, dtype=np.float32)
#       in_ = in_[:,:,::-1]
#       in_ -= self.mean  

       return im


    def load_label_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) )
        
        im = Image.open('{}/{}'.format(self.davis_dir, idx))
        im    = im.resize((img_W,img_H))
        im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
        im    = im.rotate(rotation)
        if flip:
           im = im.transpose(Image.FLIP_LEFT_RIGHT)

        if scale>0:
#           w_start = np.int(random.random()*(img_W - self.W))
#           h_start = np.int(random.random()*(img_H - self.H))
#           box     = (w_start, h_start, w_start+self.W, h_start+self.H)
           box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
           im      = im.crop(box)
        else:
           im  = im.resize((self.W, self.H))

        if self.scale != 1:
           im  = im.resize((np.int(self.W*self.scale), np.int(self.H*self.scale)))

        label = np.array(im, dtype=np.uint8)
        print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        
        return label


    def get_bboxes(self, label, box_num):
        label = np.array(label, dtype=np.uint8)
        pos   = np.where(label > 0 )
        rois     = np.zeros((box_num, 5))
        bb     = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
        for k in range(box_num):
           rois[k, 1:5] = bb
        return rois


    def get_bbox(self, label):
        label = np.array(label, dtype=np.uint8)
        pos   = np.where(label > 0 )
        if len(pos[0])<1024:
           print >> sys.stderr, 'Escape very small object (area < 1024).'
           bb = [0, 0, 0, 0]
           return bb
        else:
           bb     = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
           print>>sys.stderr, 'gt = {}; area = {};'.format(bb, len(pos[0]))
        return bb


    def get_rois(self, label, box_num, fg_rate):
        label = np.array(label, dtype=np.uint8)
        pos   = np.where(label > 0)
        W        = label.shape[1]
        H        = label.shape[0]    
        print>>sys.stderr, label.shape
        rois     = np.zeros((box_num, 5))
        gtbb     = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
        box      = np.zeros_like(gtbb)
        bb       = np.zeros((1,5))
        k        = 0
        while k < np.int(box_num*fg_rate):
              bb[0, 1]    = random.randint(0,W)
              bb[0, 2]    = random.randint(0,H)
              bb[0, 3]    = random.randint(bb[0, 1],W)
              bb[0, 4]    = random.randint(bb[0, 2],H)
              if (bb[0,4] - bb[0,2]+1)*(bb[0,3]-bb[0,1]+1) < 8*8*4:
                 continue
              iou         = self.func_iou(bb[0, 1:5], gtbb)
#              print>>sys.stderr, 'gt = {}; proposal = {}; IoU = {};'.format(gtbb, bb, iou)
              if iou > self.fg_thre:
                 rois[k,...] = bb
                 k = k + 1
#                 print>>sys.stderr, 'box_num = {}; gt = {}; proposal = {}; IoU = {};'.format(k, gtbb, bb[0, 1:5], iou)
          
        while k < box_num:
              bb[0, 1]    = random.randint(0,W)
              bb[0, 2]    = random.randint(0,H)
              bb[0, 3]    = random.randint(bb[0, 1],W)
              bb[0, 4]    = random.randint(bb[0, 2],H)
              if (bb[0,4] - bb[0,2]+1)*(bb[0,3]-bb[0,1]+1) < 8*8*4:
                 continue
              iou         = self.func_iou(bb[0, 1:5], gtbb)
#             print>>sys.stderr, 'obj = {}; gt = {}; proposal = {}; IoU = {};'.format(obj_id, gtbb, bb, iou)
              if iou < 0.5:
                 rois[k,...] = bb
                 k = k + 1

        return rois



    def func_iou(self, bb, gtbb):
        iou = 0
        iw = min(bb[2],gtbb[2]) - max(bb[0],gtbb[0]) + 1
        ih = min(bb[3],gtbb[3]) - max(bb[1],gtbb[1]) + 1
        if iw>0 and ih>0:
                ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + (gtbb[2]-gtbb[0]+1)*(gtbb[3]-gtbb[1]+1) - iw*ih
                iou = np.float32(iw*ih*1.0/ua)

        return iou

    def get_img_rois(self, img, rois):
        #img      = Image.fromarray(img)
        img_rois = np.zeros((self.box_num, 3, self.pool_h, self.pool_w))
        for i in range(self.box_num):
            bb   = rois[i,1:5]
            ROI  = img.crop(bb)
            ROI  = ROI.resize((np.int(self.pool_w), np.int(self.pool_h)))
      #      print>>sys.stderr, ROI.shape
            in_  = np.array(ROI, dtype=np.float32)
            in_  = in_[:,:,::-1]
            in_ -= self.mean
            ROI  = in_.transpose((2,0,1))
            img_rois[i,...] = ROI

        return img_rois


    def get_lab_rois(self, lab, rois):
        lab      = Image.fromarray(lab)
        lab_rois = np.zeros((self.box_num, self.pool_h, self.pool_w))
        for i in range(self.box_num):
            bb   = rois[i,1:5]
            ROI  = lab.crop(bb)
            ROI  = ROI.resize((np.int(self.pool_w), np.int(self.pool_h)))
            ROI  = np.array(ROI, dtype=np.uint8)
            ROI  = np.uint8(ROI>0)
            lab_rois[i,...] = ROI

        return lab_rois


    def calculate_weight_rois(self, labels):
        weights = np.ones((self.box_num, self.pool_h, self.pool_w))
        for i in range(self.box_num):
            label = labels[i,...]
            pos = np.where(label==1)
            neg = np.where(label==0)
            weight_pos = len(pos[0])*1.0/(len(pos[0])+len(neg[0]))
            for k in range(len(pos[0])):
                weights[i, pos[0][k],pos[1][k]] = 1 - weight_pos
#            print >> sys.stderr, 'pos_num = {}, neg_num = {}, weight_pos = {}'.format(len(pos[0]), len(neg[0]), 1 - weight_pos)

        return weights

