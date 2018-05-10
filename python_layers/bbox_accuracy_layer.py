import caffe
import numpy as np
import math
from scipy.misc import imresize
import sys
import random

class BBoxAccLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two bottoms: score label")


    def reshape(self, bottom, top):
        #top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1],bottom[0].data.shape[2],bottom[0].data.shape[3])I
        #top[1].reshape(bottom[1].data.shape[0],bottom[1].data.shape[1],bottom[1].data.shape[2],bottom[1].data.shape[3])
        self.score  = np.zeros_like(bottom[0].data, dtype=np.float32) 
        self.label  = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.H      = 480
        self.W      = 854
       


    def forward(self, bottom, top):
        self.score  = bottom[0].data
        self.label  = bottom[1].data
        label = self.label
        num_bg = 0
        num_fg = 0
        correct_bg = 0
        correct_fg = 0
        for iw in range(self.W):
             for ih in range(self.H):
                  if label[0,0,ih,iw] == -1:
                     continue
                  elif self.score[0,0,ih,iw]>self.score[0,1,ih,iw]:
                     if label[0,0,ih,iw] == 0:
                        num_bg     = num_bg + 1
                        correct_bg = correct_bg + 1
                     else:
                        num_fg     = num_fg + 1
                  else:
                     if label[0,0,ih,iw] == 1:
                        num_fg     = num_fg + 1
                        correct_fg = correct_fg + 1
                     else:
                        num_bg     = num_bg + 1

        if num_bg > 0 and num_fg > 0:
           print >> sys.stderr,'pr_bg = {}; pr_fg = {}; pr_all = {}'.format(correct_bg*1.0/num_bg,correct_fg*1.0/num_fg,(correct_bg+correct_fg)*1.0/(num_bg+num_fg))


    def backward(self, top, propagate_down, bottom):
        pass
