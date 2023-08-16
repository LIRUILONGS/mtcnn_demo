import sys
import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F


import hopenet
import hopenet_utils
from mtcnn import MTCNN
import cv2

from PIL import Image


class HopenetFace:

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self,snapshot_path ):
        snapshot_path
        # ResNet50 structure
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        # Load snapshot
        saved_state_dict = torch.load(
            snapshot_path, map_location=torch.device('cpu'))
        model.load_state_dict(saved_state_dict)
        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(
                                                       224), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # Test the Model
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        self.model = model
        self.total = 0
        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor)
        self.yaw_error = .0
        self.pitch_error = .0
        self.roll_error = .0
        self.l1loss = torch.nn.L1Loss(size_average=False)

    def headPosEstimate(self, img):
        """
        @Time    :   2023/08/15 22:10:20
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   头部姿态检测
                     Args:
                       
                     Returns:
                       void
        """
        
        img = img.convert('RGB')
        cv2_img = np.asarray(img)
        # print(cv2_img.shape)
        cv2_img = cv2.resize(cv2_img, (224, 224))[:, :, ::-1]
        cv2_img = cv2_img.astype(np.uint8).copy()
        img = self.transformations(img)
        img = img.unsqueeze(0)
        images = Variable(img)
        yaw, pitch, roll = self.model(images)
        # Binned predictions
        _, self.yaw_bpred = torch.max(yaw.data, 1)
        _, self.pitch_bpred = torch.max(pitch.data, 1)
        _, self.roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = hopenet_utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = hopenet_utils.softmax_temperature(pitch.data, 1)
        roll_predicted = hopenet_utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(
            yaw_predicted * self.idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(
            pitch_predicted * self.idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(
            roll_predicted * self.idx_tensor, 1).cpu() * 3 - 99

        pitch = pitch_predicted[0]
        yaw = -yaw_predicted[0]
        roll = roll_predicted[0]
        #print("pitch,yaw,roll", pitch.item(), yaw.item(), roll.item())
        hopenet_utils.draw_axis(
            cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
        p = pitch.item()
        y = yaw.item()
        r = roll.item()
        return cv2_img,p,y,r
    
