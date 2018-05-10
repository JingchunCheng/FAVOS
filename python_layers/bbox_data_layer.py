import caffe
import numpy as np
from numpy import *
import math
from scipy.misc import imresize
import sys
import random

class BboxDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need three bottoms: label, rois")
        if len(top) != 1:
            raise Exception("Need one top: bbox_targets")


    def reshape(self, bottom, top):
        self.label  = np.zeros_like(bottom[0].data, dtype=np.float32) 
        self.rois   = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.bbox_targets = np.zeros([bottom[1].data.shape[0], 4], dtype=np.float32)
        top[0].reshape(self.rois.shape[0],4)

    def forward(self, bottom, top):
        self.label   = bottom[0].data
        self.label      = self.label.reshape([self.label.shape[1],self.label.shape[2]])
        self.rois    = bottom[1].data
        self.box_num = self.rois.shape[0]
        gt_box       = self.get_bbox(self.label)
        
        for i in range(self.box_num):
              bbgt  = self.transform_box(gt_box, self.rois[i,1:5])
              bbgt  = bbgt.reshape(1,4)
              self.bbox_targets[i,...] = bbgt
           

       
        top[0].reshape(self.box_num,4)      
        top[0].data[...] = self.bbox_targets
 


    def backward(self, top, propagate_down, bottom):
        pass


    def get_bbox(self, label):
        label = np.array(label, dtype=np.uint8)
        pos   = np.where(label == 1)
        bb     = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
        #print>>sys.stderr, 'gt = {}; area = {};'.format(bb, len(pos[0]))
        return bb


    def transform_box(self,box,roi):
        x_a = (roi[0] + roi[2])*1.0/2
        y_a = (roi[1] + roi[3])*1.0/2
        w_a = (roi[2]-roi[0]+1)*1.0
        h_a = (roi[3]-roi[1]+1)*1.0

        bb  = zeros([4,1],dtype=np.float32)
        bb[0] = (box[0] - x_a)*1.0/w_a
        bb[1] = (box[1] - y_a)*1.0/h_a
        bb[2] = box[2] - box[0] + 1
        bb[2] = math.log(bb[2]*1.0/w_a)
        bb[3] = box[3] - box[1] + 1
        bb[3] = math.log(bb[3]*1.0/h_a)

        #print >> sys.stderr, 'box = {}, box_transform = {}'.format(box, bb)
        return bb
