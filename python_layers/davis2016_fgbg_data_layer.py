import caffe

import numpy as np
from PIL import Image

import cv2
from scipy.misc import imresize
from scipy.misc import imrotate

import sys

import random

class DAVIS2016FgBgDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
   """

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.davis_dir = params['davis_dir']
        self.split     = params['split']
        self.mean      = np.array(params['mean'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        self.augment    = params.get('with_augmentation', True)
        self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 480
        self.W         = 854


        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define two tops: data label weight")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/480p/{}.txt'.format(self.davis_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = -1 # we pick idx in reshape

        if self.augment:
           self.aug_num         = np.int(self.aug_params[0])
           self.max_scale       = self.aug_params[1]
           self.max_rotate      = self.aug_params[2]
           self.max_transW      = self.aug_params[3]
           self.max_transH      = self.aug_params[4]
           self.flip            = (self.aug_params[5]>0)


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
                self.idx = random.randint(0, len(self.indices)-1)
            else:
                self.idx += 1
                if self.idx == len(self.indices):
                    self.idx = 0
           

            if self.idx == (len(self.indices) - 1):
                continue
        
            idx = self.idx

            if self.augment == False or random.randint(0, self.aug_num) == 0:
               self.img    = self.load_image(self.indices[idx].split(' ')[0])
               self.label  = self.load_label(self.indices[idx].split(' ')[1])
               self.img    = imresize(self.img,    size=(self.H, self.W), interp="bilinear")
               self.label  = imresize(self.label,  size=(self.H, self.W), interp="nearest")
            else:
               scale       =  (random.random()*2-1) * self.max_scale
               rotation    =  (random.random()*2-1) * self.max_rotate
               trans_w     =  np.int( (random.random()*2-1) * self.max_transW * self.W )
               trans_h     =  np.int( (random.random()*2-1) * self.max_transH * self.H )
               if self.flip:
                  flip     = (random.randint(0,1) > 0)
               else:
                  flip     = False
               self.img    = self.load_image_transform(self.indices[idx].split(' ')[0], scale, rotation, trans_h, trans_w, flip)
               self.label  = self.load_label_transform(self.indices[idx].split(' ')[1], scale, rotation, trans_h, trans_w, flip)


            if self.scale != 1:
               self.img   = imresize(self.img,    size=(np.int(self.H*self.scale), np.int(self.W*self.scale)), interp="bilinear")
               self.label = imresize(self.label,  size=(np.int(self.H*self.scale), np.int(self.W*self.scale)), interp="nearest")


            self.weight = self.calculate_weight(self.label)
            self.img    = self.img.transpose((2,0,1))
            self.label  = self.label[np.newaxis, ...]
            self.weight = self.weight[np.newaxis, ...]
            break            

        # reshape tops to fit (leading 2 is for batch dimension)
        top[0].reshape(1, *self.img.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, *self.weight.shape)
   
    def forward(self, bottom, top):
        top[0].data[...] = self.img
        top[1].data[...] = self.label
        top[2].data[...] = self.weight

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
        print >> sys.stderr, 'loading Original {}'.format(idx)
        im = Image.open('{}/{}'.format(self.davis_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/{}'.format(self.davis_dir, idx))
        label = np.array(im>0, dtype=np.uint8)
        print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
    
        #print(label)
        return label

    def load_image_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
       img_W = np.int( self.W*(1.0 + scale) )
       img_H = np.int( self.H*(1.0 + scale) ) 

       print >> sys.stderr, 'loading {}'.format(idx)
       print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

       im    = Image.open('{}/{}'.format(self.davis_dir, idx))
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
       

       in_ = np.array(im, dtype=np.float32)
       in_ = in_[:,:,::-1]
       in_ -= self.mean  

       return in_


    def load_label_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) )
        

        im    = Image.open('{}/{}'.format(self.davis_dir, idx))
        im    = im.resize((img_W,img_H))
        im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
        im    = im.rotate(rotation)
        if flip:
           im = im.transpose(Image.FLIP_LEFT_RIGHT)

        if scale>0:
           w_start = np.int(random.random()*(img_W - self.W))
           h_start = np.int(random.random()*(img_H - self.H))
           box     = (w_start, h_start, w_start+self.W, h_start+self.H)
           im      = im.crop(box)
        else:
           im  = im.resize((self.W, self.H))

        label = np.array(im>0, dtype=np.uint8)
        print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        
        
        return label



    def calculate_weight(self, annt):
        weight = np.zeros_like(annt, dtype = np.float32) + 1
        pos = np.where(annt==1)
        neg = np.where(annt==0)
        pos_weight = 1 - np.float32(len(pos[0]))/np.float32(len(pos[0])+len(neg[0]))
        weight = weight - pos_weight
        print >> sys.stderr, 'pos_weight: {}'.format(pos_weight)

        for idx  in range (len(pos[0])):
            weight[pos[0][idx], pos[1][idx]] = pos_weight

        #print(weight)
        return weight
