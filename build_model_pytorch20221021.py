# -*- coding: utf-8 -*-
#! /home/tme/anaconda3/envs/gyf_pt1.10py3.9/bin/python

'''
title = 'build_model_pytorch20111021'
author = 'Dafei'
time = '2022-10-21'
'''

import torch
import numpy as np
import cv2
import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class pytorch_Net(nn.Module):
    def __init__(self):
        super(pytorch_Net, self).__init__()

        # 第一次卷积
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，
            # 卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 48, 48, 1),
            # output:(bitch_size, 48, 48, 32), ?(48-3+2*1)/1+1 = 48
            # 卷积层
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # 数据归一化处理，使得数据在Relu之前不会因为数据过大而导致网络性能不稳定
            # 做归一化让数据形成一定区间内的正态分布
            # 不做归一化会导致不同方向进入梯度下降速度差距很大
            nn.BatchNorm2d(num_features=32),  # 归一化可以避免出现梯度散漫的现象，便于激活。
            nn.ReLU(inplace=True)  # 激活函数
        )
        # print("conv1:",torch.cuda.memory_allocated())
        # print("conv1 max:", torch.cuda.max_memory_allocated())
        # print("memory_summary():", torch.cuda.memory_summary())

        # 第二次卷积
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 48, 48, 32),
            # output:(bitch_size, 46, 46, 32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True) # 激活函数
        )
        # print("memory_summary():", torch.cuda.memory_summary())

        #第三次卷积
        self.conv3 = nn.Sequential(
            #input:(bitch_size, 46, 46, 32),
            #output:(bitch_size, 46, 46, 64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        # print("memory_summary():", torch.cuda.memory_summary())

        #池化
        self.maxpool = nn.Sequential(
            nn.Dropout(p=0.5),
            # 最大值池化
            nn.MaxPool2d(kernel_size= 3), # output(bitch_size, 46, 46, 64)

        )
        # print("memory_summary():", torch.cuda.memory_summary())

        #全连接层
        self.fc = nn.Sequential(

            nn.Linear(in_features=2097152, out_features=64), #2097152 / 1032192
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=7),
            nn.Softmax()

        )
        # print("memory_summary():", torch.cuda.memory_summary())

    def forward(self, x):
        x = self.conv1(x)
        # print("conv1 memory_summary():", torch.cuda.memory_summary())
        x = self.conv2(x)
        # print("conv2 memory_summary():", torch.cuda.memory_summary())
        x = self.conv3(x)
        # print("conv3 memory_summary():", torch.cuda.memory_summary())
        x = self.maxpool(x)
        # print("maxpool memory_summary():", torch.cuda.memory_summary())

        # 数据扁平化
        x = torch.flatten(x)  # 输出维度，-1表示该维度自行判断
        # print("flatten memory_summary():", torch.cuda.memory_summary())
        y = self.fc(x)
        # print("fc memory_summary():", torch.cuda.memory_summary())

        return y


