import caffe

import numpy as np
from PIL import Image
from scipy.misc import imresize

import sys

import random

from scipy import ndimage



class DavisSiameseTestScoreLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.davis_dir = params['davis_dir']
        self.split     = params['split']
        self.mean      = np.array(params['mean'])
        self.sc_params = np.array(params['aug_params']) # min_scale, max_scale, stride, flag_search_range (0: whole image, 1: search_range), search_range 
        self.H         = 480
        self.W         = 854

       
        self.min_scale         = self.sc_params[0]
        self.max_scale         = self.sc_params[1]
        self.stride            = self.sc_params[2]
        self.flag_range        = (self.sc_params[3]>0)
        self.search_range      = self.sc_params[4]
        # anchors: 1*2, 2*1, 1*1

        # tops
        if len(top) != 3:
            raise Exception("Need to define three tops: scores, scales, anchors")
        # data layers have no bottoms
        if len(bottom) != 3:
            raise Exception("Need two bottoms: gt1_box, feat1, feat2")

    def reshape(self, bottom, top):
        self.bbox    =  np.zeros_like(bottom[0].data, dtype = np.float32)
        self.feat1   =  np.zeros_like(bottom[1].data, dtype = np.float32)
        self.feat2   =  np.zeros_like(bottom[2].data, dtype = np.float32)       
        self.num_scoremaps = 3.0*( (self.max_scale - self.min_scale)*1.0/self.stride + 1 )
        self.scales     = np.zeros([self.num_scoremaps, 1], dtype = np.float32)
        self.anchors    = np.zeros([self.num_scoremaps, 2], dtype = np.float32)
        self.score_maps = np.zeros([self.num_scoremaps, bottom[1].shape[2], bottom[1].shape[3]],  dtype = np.float32)
        # reshape tops
        top[0].reshape(self.num_scoremaps, bottom[1].shape[2], bottom[1].shape[3])
        top[1].reshape(self.num_scoremaps, 1)
        top[2].reshape(self.num_scoremaps, 2)


    def forward(self, bottom, top):
        self.bbox  = bottom[0].data
        self.feat1 = bottom[1].data
        self.feat2 = bottom[2].data

        self.feat1 = self.feat1.reshape([self.feat1.shape[1], self.feat1.shape[2], self.feat1.shape[3]])
        self.feat1 = self.feat1.transpos((2,3,1))
        feat_bbox  = self.extract_feat_ROI(self.feat1, self.bbox) # get gt features
        
        self.feat2 = self.feat2.reshape([self.feat1.shape[1], self.feat1.shape[2], self.feat1.shape[3]])
        self.feat2 = self.feat2.transpos((2,3,1))

        anchor = [1, 1]
        for i in range(np.int(self.num_scoremaps/3)):
            scale     = self.min_scale + i*self.stride
            feat_map  = self.extract_feat_map(self.feat2, scale, anchor) # get scaled feature maps
            score_map = ndimage.convolve(feat_bbox, feat_map, mode='reflect', cval=0.0)
            self.score_maps[i,...] = score_map
            self.scales[i]         = scale
            self.anchors[i,...]    = anchor


        # assign output
        top[0].data[...] = self.score_maps
        top[1].data[...] = self.scales
        top[2].data[...] = self.anchors
 


    def backward(self, top, propagate_down, bottom):
        pass


    def extract_feat_ROI(self, feat, bbox):
        bbox = np.round(bbox)
        feat_bbox = feat[bbox[0]:bbox[2],bbox[1]:bbox[3],...]

        return feat_bbox


    def extract_feat_map(self, feat, scale, anchor):
        

        
