import caffe
import numpy as np
from numpy import *
import math
from scipy.misc import imresize
import sys
import random

class SimilarityLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two bottoms: bbox, feats")


    def reshape(self, bottom, top):
        self.bbox   = np.zeros_like(bottom[0].data, dtype=np.float32) 
        self.feats  = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.H      = 480
        self.W      = 854
        top[0].reshape(self.feats.shape[0],self.bbox.shape[0])


    def forward(self, bottom, top):
        self.feats  = bottom[1].data
        self.bbox   = bottom[0].data
        num_obj     = self.bbox.shape[0]
        self.feat1  = self.feats[0:num_obj,...] 
        self.feat2  = self.feats[num_obj:self.feats.shape[0],...]
        #print self.feat1.shape 300,4096,1,1
        #print >> sys.stderr, self.feat1                 
        #print >> sys.stderr, self.feat2
        print >> sys.stderr, 'Number of objects: {}'.format(num_obj)
        
        dist = np.zeros([self.feat2.shape[0], num_obj], dtype=np.float32)
        for obj_id in range(num_obj):
            for i in range(self.feat2.shape[0]): 
                vector1 = np.float32(mat(self.feat1[obj_id,...,0]))
                vector2 = np.float32(mat(self.feat2[i,...,0]))
                dist[i, obj_id] = sqrt(((vector1-vector2).T)*(vector1-vector2))       
  
        top[0].reshape(self.feat2.shape[0], num_obj)
        top[0].data[...] = dist

    def backward(self, top, propagate_down, bottom):
        pass
