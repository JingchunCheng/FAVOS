import caffe
import numpy as np
from numpy import *
import math
from scipy.misc import imresize
import sys
import random

class ResValueLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one bottom.")


    def reshape(self, bottom, top):
        self.response  = np.zeros_like(bottom[0].data, dtype=np.float32) 

    def forward(self, bottom, top):
        self.response  = bottom[0].data
        print >> sys.stderr,'Response Absolute Value Mean = {}; Response Value Range = [{}, {}];'.format(np.mean(np.abs(self.response)), np.min(self.response), np.max(self.response))


    def backward(self, top, propagate_down, bottom):
        pass
