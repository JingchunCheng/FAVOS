import caffe
import numpy as np
from numpy import *
import math
from scipy.misc import imresize
import sys
import random

from PIL import Image

class MaskDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two bottoms: label, rois")
        if len(top) != 2:
            raise Exception("Need two top: mask_targets, weights")
        params      = eval(self.param_str)
        self.mask_w = params.get('mask_w', 14)
        self.mask_h = params.get('mask_h', 14)


    def reshape(self, bottom, top):
        self.label  = np.zeros_like(bottom[0].data, dtype=np.float32) 
        self.rois   = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.mask_targets = np.zeros([self.rois.shape[0], self.mask_h, self.mask_w],dtype=np.uint8)
        self.weights      = np.zeros([self.rois.shape[0], self.mask_h, self.mask_w],dtype=np.float32)
        top[0].reshape(self.rois.shape[0], self.mask_h, self.mask_w)
        top[1].reshape(self.rois.shape[0], self.mask_h, self.mask_w)

    def forward(self, bottom, top):
        self.label  = bottom[0].data
        self.label  = self.label.reshape(self.label.shape[1], self.label.shape[2])
#        print >> sys.stderr, self.label
        self.rois   = bottom[1].data
        box_num     = self.rois.shape[0]
        label_map   = self.label
        for i in range(box_num):
              label_map_new = self.label_in_box(label_map, self.rois[i,...])
              label_map_new = imresize(label_map_new,  size=(np.int(self.mask_h), np.int(self.mask_w)), interp="nearest", mode='F')
              label_map_new = np.array(label_map_new, dtype=np.uint8)
 #             print label_map_new
 #             pos   = np.where(label_map_new == 1)
 #             print >> sys.stderr, 'Number of positive pixels after resize: {}'.format(len(pos[0]))
              self.mask_targets[i,...] = label_map_new
              self.weights[i,...]      = self.calculate_weight(label_map_new)
 #       print self.mask_targets
        top[0].data[...] = self.mask_targets
        top[1].data[...] = self.weights
 

    def backward(self, top, propagate_down, bottom):
        pass


    def label_in_box(self, label, bb):
        bb = bb[1:5]
        label = Image.fromarray(label)
#        print >>sys.stderr, bb
        label = label.crop(bb)
#        label = np.array(label, dtype=np.uint8)
#        pos   = np.where(label == 1)
#        print >> sys.stderr, 'Number of positive pixels: {}'.format(len(pos[0]))

        return label



    def calculate_weight(self, label):
        pos = np.where(label==1)
        neg = np.where(label==0)
        weight_pos = len(pos[0])*1.0/(len(pos[0])+len(neg[0]))
        weight = np.ones_like(label, dtype=np.float32)*weight_pos
        for i in range(len(pos[0])):
            weight[pos[0][i],pos[1][i]] = 1 - weight_pos

        print >> sys.stderr, 'pos_num = {}, neg_num = {}, weight_pos = {}'.format(len(pos[0]), len(neg[0]), 1 - weight_pos)

        return weight
