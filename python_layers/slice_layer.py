import caffe
import numpy as np
from numpy import *
import math
from scipy.misc import imresize
import sys
import random

class SliceLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one bottom: feat_concat")


    def reshape(self, bottom, top):
        self.feats   = np.zeros_like(bottom[0].data, dtype=np.float32) 
        top[0].reshape(self.feats.shape[0]-1,self.feats.shape[1],self.feats.shape[2],self.feats.shape[3])
        top[1].reshape(self.feats.shape[0]-1,self.feats.shape[1],self.feats.shape[2],self.feats.shape[3])


    def forward(self, bottom, top):
        self.feats  = bottom[0].data
        num_obj     = self.feats.shape[0]-1
        self.feat2  = self.feats[1:self.feats.shape[0],...]
        self.feat1  = np.zeros_like(self.feat2,dtype=np.float32)
        
        print >> sys.stderr, 'Number of objects: {}'.format(num_obj)
        
        for obj_id in range(num_obj):
            self.feat1[obj_id,...] = self.feats[0,...] 
  
        top[0].reshape(*self.feat1.shape)
        top[0].data[...] = self.feat1
        top[1].reshape(*self.feat2.shape)
        top[1].data[...] = self.feat2


    def backward(self, top, propagate_down, bottom):
        pass
