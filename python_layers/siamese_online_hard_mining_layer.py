import caffe
import numpy as np
from numpy import *
import math
from scipy.misc import imresize
import sys
import random

class SiaHardMiningLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two bottoms: label, feat1, feat2")


    def reshape(self, bottom, top):
        self.label      = np.zeros_like(bottom[0].data, dtype=np.float32) 
        self.label_new  = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.feat1      = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.feat2      = np.zeros_like(bottom[2].data, dtype=np.float32)
        self.maintain_rate =  0.5
        self.H          = 480
        self.W          = 854
        self.pos        = np.zeros([ 2,self.label.shape[0] ])
        top[0].reshape(bottom[0].shape[0]/2, 1)
        top[1].reshape(bottom[1].shape[0]/2, bottom[1].shape[1], bottom[1].shape[2], bottom[1].shape[3])
        top[2].reshape(bottom[1].shape[0]/2, bottom[1].shape[1], bottom[1].shape[2], bottom[1].shape[3])


    def forward(self, bottom, top):
        self.label  = bottom[0].data
        self.feat1  = bottom[1].data
        self.feat2  = bottom[2].data
        label       = self.label
        #print self.feat1.shape 300,4096,1,1
        #print >> sys.stderr, self.feat1                 
        #print >> sys.stderr, self.feat2
        
        dist = np.zeros([len(self.label),1], dtype=np.float32)
        for i in range(len(self.label)): 
            vector1 = np.float32(mat(self.feat1[i,...,0]))
            vector2 = np.float32(mat(self.feat2[i,...,0]))
            dist[i] = sqrt(((vector1-vector2).T)*(vector1-vector2))       


        pos   = np.where(self.label == 1)
        D_sim = dist[pos]       
        thre  = np.median(D_sim)
        for i in range(len(pos[0])):
            if D_sim[i] < thre:
               label[pos[0][i]] = -1  

        pos   = np.where(self.label == 0)
        D_diff= dist[pos]
        thre  = np.median(D_diff)
        for i in range(len(pos[0])):
            if D_diff[i] > thre:
               label[pos[0][i]] = -1  

#        print>> sys.stderr, self.label    

        pos1 = np.where(label == 1)
        pos2 = np.where(label == 0)
        if len(pos1[0])>len(pos2[0]):
           pos = np.zeros_like(pos2)
           pos[0][...] = pos1[0][0:len(pos2[0])]
           pos[1][...] = pos1[1][0:len(pos2[0])]
           pos1 = pos

        if len(pos2[0])>len(pos1[0]):
           pos = np.zeros_like(pos1)
           pos[0][...] = pos2[0][0:len(pos1[0])]
           pos[1][...] = pos2[1][0:len(pos1[0])]
           pos2 = pos


        self.pos       = np.concatenate((pos1, pos2), axis = 1)
        self.label_new = self.label[self.pos[0]]
        self.feat1     = self.feat1[self.pos[0],...]
        self.feat2     = self.feat2[self.pos[0],...]
#        print>> sys.stderr,  self.label_new.shape
       
        top[0].reshape(*self.label_new.shape)
        top[0].data[...] = self.label_new
        top[1].reshape(*self.feat1.shape)
        top[1].data[...] = self.feat1
        top[2].reshape(*self.feat2.shape)
        top[2].data[...] = self.feat2



    def backward(self, top, propagate_down, bottom):
        for i in range(3):
            if not propagate_down[i]:
               continue

            if i == 0:
               continue

            bottom[i].diff[...] = np.zeros_like(bottom[i].data, dtype=np.float32)
#            print >> sys.stderr, top[i].diff.shape
#            print >> sys.stderr, bottom[i].diff.shape
            for j in range(len(self.pos[0])):
                bottom[i].diff[self.pos[0][j],...] += top[i].diff[j,...] 
              

