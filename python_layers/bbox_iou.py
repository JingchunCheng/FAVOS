import caffe
import numpy as np
from numpy import *
import math
from scipy.misc import imresize
import sys
import random
import math

class BBoxIoULayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two bottoms: label, roi, bbox_pred")


    def reshape(self, bottom, top):
        self.label      = np.zeros_like(bottom[0].data, dtype=np.float32) 
        self.roi        = np.zeros_like(bottom[1].data, dtype=np.float32)
        self.bbox_pred  = np.zeros_like(bottom[2].data, dtype=np.float32)
       


    def forward(self, bottom, top):
        self.label      = bottom[0].data
        self.label      = self.label.reshape([self.label.shape[1],self.label.shape[2]])
        self.rois       = bottom[1].data
        self.bbox_pred  = bottom[2].data
        box_num         = self.rois.shape[0]
        gt_box          = self.get_bbox(self.label)
#        print >> sys.stderr, self.label.shape
        
        IoUs = np.zeros([box_num, 2], dtype=np.float32)
        for i in range(box_num): 
            roi = self.rois[i,1:5]
            box = self.bbox_pred[i,...,0,0]
            box = self.recover_box(box,roi)
            iou_roi  = self.func_iou(roi, gt_box)
            iou_pred = self.func_iou(box, gt_box) 
            IoUs[i,0]= iou_roi
            IoUs[i,1]= iou_pred

#        print >> sys.stderr, 'gt = {}'.format(gt_box)
        ave_iou = np.sum(IoUs,  axis=0)*1.0/box_num

        print >> sys.stderr,'IoU before bbrg = {}; IoU after bbrg = {};'.format(ave_iou[0], ave_iou[1])


    def backward(self, top, propagate_down, bottom):
        pass


    def func_iou(self, bb, gtbb):
        iou = 0 
        iw = min(bb[2],gtbb[2]) - max(bb[0],gtbb[0]) + 1 
        ih = min(bb[3],gtbb[3]) - max(bb[1],gtbb[1]) + 1 
        if iw>0 and ih>0:
                ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + (gtbb[2]-gtbb[0]+1)*(gtbb[3]-gtbb[1]+1) - iw*ih 
                iou = np.float32(iw*ih*1.0/ua)

        return iou 



    def get_bbox(self, label):
        label = np.array(label, dtype=np.uint8)
        pos   = np.where(label == 1)
        bb     = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
        print>>sys.stderr, 'gt = {}; area = {};'.format(bb, len(pos[0]))
        return bb


    def recover_box(self,box,roi):
        x_a = (roi[0] + roi[2])*1.0/2
        y_a = (roi[1] + roi[3])*1.0/2
        w_a = (roi[2]-roi[0]+1)*1.0
        h_a = (roi[3]-roi[1]+1)*1.0

        x   = box[0]*w_a + x_a
        y   = box[1]*h_a + y_a
        w   = math.exp(box[2])*w_a
        h   = math.exp(box[3])*h_a

        bb  = zeros([4,1],dtype=np.float32)
        bb[0] = x - w*1.0/2
        bb[1] = y - h*1.0/2
        bb[2] = x + w*1.0/2
        bb[3] = y + h*1.0/2

#        print >> sys.stderr, 'box_transform = {}, box = {}'.format(box, bb)
        return bb

