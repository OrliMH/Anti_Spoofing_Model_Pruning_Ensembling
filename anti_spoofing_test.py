
# coding:utf-8
import dlib
import numpy as np
from copy import deepcopy
import cv2
import os
import torch 
from imgaug import augmenters as iaa
import random 
from torch.utils.data import *
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torch.nn as nn
import math 
from torch.nn.modules.distance import PairwiseDistance
RESIZE_SIZE = 112
class FaceSpoofing(object):
    def __init__(self, channels=3, image_size=32):
        super(FaceSpoofing, self).__init__
        self.channels = channels
        self.image_size = image_size 
    def TTA_36_cropps(self, image, target_shape=(32, 32, 3)): # 返回固定位置的36个
        image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

        width, height, d = image.shape
        target_w, target_h, d = target_shape

        start_x = ( width - target_w) // 2
        start_y = ( height - target_h) // 2

        starts = [[start_x, start_y],# 取固定位置的9块

                [start_x - target_w, start_y],
                [start_x, start_y - target_w],
                [start_x + target_w, start_y],
                [start_x, start_y + target_w],

                [start_x + target_w, start_y + target_w],
                [start_x - target_w, start_y - target_w],
                [start_x - target_w, start_y + target_w],
                [start_x + target_w, start_y - target_w],
                ]

        images = []

        for start_index in starts:
            image_ = image.copy()
            x, y = start_index

            if x < 0:
                x = 0
            if y < 0:
                y = 0

            if x + target_w >= RESIZE_SIZE:
                x = RESIZE_SIZE - target_w-1 # w比具体像素点个数多1
            if y + target_h >= RESIZE_SIZE:
                y = RESIZE_SIZE - target_h-1

            zeros = image_[x:x + target_w, y: y+target_h, :]

            image_ = zeros.copy()

            zeros = np.fliplr(zeros)
            image_flip_lr = zeros.copy()

            zeros = np.flipud(zeros)
            image_flip_lr_up = zeros.copy()

            zeros = np.fliplr(zeros)
            image_flip_up = zeros.copy()

            images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
            images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
            images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
            images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

        return images # 9*4, 1, h, w, c ==> 36, 1, 32, 32, 3
    def color_augumentor(self, image, target_shape=(32, 32, 3)):# target_shape传进来128
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])
        image =  augment_img.augment_image(image)
        image = self.TTA_36_cropps(image, target_shape)
        return image # 36, 1, 32, 32, 3
    
    def get_input(self, face_align):
        face_align_aug = self.color_augumentor(face_align)  # 36, 1, 32, 32, 3
        n = len(face_align_aug)
        face_align_aug = np.concatenate(face_align_aug, axis=0) # (36, 32, 32, 3)
        face_align_aug = np.transpose(face_align_aug, (0, 3, 1, 2)) # 36 3 32 32
        face_align_aug = face_align_aug.astype(np.float32)
        face_align_aug = face_align_aug.reshape([1, n, self.channels, self.image_size, self.image_size]) # 1, 36 3 32 32
        face_align_aug = face_align_aug / 255.0

        return torch.FloatTensor(face_align_aug)

    # 实现活体检测二分类
    def classify(self,face_align):
        model_pth = "/home/disk2/live_detect/week3/week3code-CVPR19-Face-Anti-spoofing/models/model_A_color_32/entire_prune_net_checkpointglobal_min_acer_model.pth"
        model = torch.load(model_pth)
        model.eval()

        input_ = self.get_input(face_align) # 1, 36 3 32 32 || b n c h w
        b,n,c,w,h = input_.size()
        input_ = input_.view(b*n,c,w,h)

        with torch.no_grad():
            logit,_,_   = model(input_)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False) # b 2
            prob = F.softmax(logit, 1) # 0 fake 1 real b 2
            one_prob = prob[0]
            if one_prob[0] > one_prob[1]:
                return False # fake
        return True 


face_spoofing = FaceSpoofing()

predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
# cur_img_pth = "/home/disk2/anti_spoofing_data/fake_part/CLKJ_AS0005/04_en_b.rssdk/color/101.jpg" # False
# cur_img_pth = "/home/disk2/live_detect/week3/week3code-CVPR19-Face-Anti-spoofing/feifeili.jpg" # True 
cur_img_pth = "/home/disk2/live_detect/week3/week3code-CVPR19-Face-Anti-spoofing/yoshua_picture.jpg" # True
frame_src = cv2.imread(cur_img_pth)
frame = frame_src
dets = detector(frame, 1)
for det in dets:
    shape=predictor(frame, det)
    face_align=dlib.get_face_chip(frame, shape, 150,0.1)

    print(face_spoofing.classify(face_align))