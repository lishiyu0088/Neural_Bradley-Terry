################
#
# Physics-informed GAN
# H. Ma
# 20201112
# Modified from "Deep Flow Prediction - N. Thuerey"
# load the images
#
################

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import re
import math
from NIMA.nima.clean_dataset import clean_and_split
import click
from pathlib import Path

def imageLoader_AVA(path_images):
    numworkers =12
    click.echo(f"Clean and split dataset to train|val|test in {numworkers} threads. It will takes several minutes")
    clean_and_split(
    path_to_ava_txt = "./images/AVA.txt",
    path_to_save_csv = "./images",
    path_to_images = path_images,
    train_size = 0.8,
    num_workers = numworkers,
    )
    click.echo("Done!")

def imageLoader_train(data, dataDir):
    folders = os.listdir(dataDir)
    folders.sort()
    data.win = []
    data.filename = []
    data.train = []
    data.train = torch.as_tensor(data.train)

    for i, folder in enumerate(folders):
        filelist = os.listdir(dataDir + "/" + folder)
        data.imgs = []
        data.score = []
        data.file = []

        for j, file in enumerate(filelist):
            if file != "0.jpg":
                npfile = cv2.imread(dataDir + "/" + folder + "/" +file)
                npfile = np.transpose(npfile, (2, 0, 1))
                data.imgs_tensor = torch.from_numpy(npfile).float()
                data.imgs_tensor = torch.unsqueeze(data.imgs_tensor, 0)
                data.imgs_small = torch.nn.functional.interpolate(data.imgs_tensor, size=(224, 224), scale_factor=None,
                                                              mode='nearest', align_corners=None)
                data.imgs.append(data.imgs_small)
                string = re.findall("\d+\.\d+", file)
                for item in string:
                    score = float(item)
                data.score.append(score)
                data.file.append(file)
        for k in range(len(data.imgs)):
            for l in range(len(data.imgs)):
                if k!= l:
                    batch = torch.cat([data.imgs[k],data.imgs[l]],1)
                    data.train = torch.cat([data.train,batch],0)
                    data.win.append(data.score[k]/(data.score[k]+data.score[l]))
                    data.filename.append([data.file[k],data.file[l]])

    data.totalLength = len(data.train)
    # data.totalLength = totalLength
    # find a picture to plot as the experimental reference
    imgs_plot = data.imgs[0][0, 0, :, :]
    # plt.imshow(imgs_plot.cpu(), cmap='Greys_r')
    plt.imshow(imgs_plot.cpu())
    plt.show()



    # print("=====================Finished===========================")

    return data

# data=[]
# imageLoader(data)

def imageLoader_vali(data, dataDir):
    folders = os.listdir(dataDir)
    folders.sort()
    data.out = []
    data.scores = []
    data.filenames = []
    data.foldername = folders

    for i, folder in enumerate(folders):
        filelist = os.listdir(dataDir + "/" + folder)
        data.score = []
        data.filename = []
        data.train = []
        data.train = torch.as_tensor(data.train)

        for j, file in enumerate(filelist):
            if file != "0.jpg":
                npfile = cv2.imread(dataDir + "/" + folder + "/" +file)
                npfile = np.transpose(npfile, (2, 0, 1))
                data.imgs_tensor = torch.from_numpy(npfile).float()
                data.imgs_tensor = torch.unsqueeze(data.imgs_tensor, 0)
                data.imgs_small = torch.nn.functional.interpolate(data.imgs_tensor, size=(224, 224), scale_factor=None,
                                                              mode='nearest', align_corners=None)
                string = re.findall("\d+\.\d+", file)
                for item in string:
                    score = float(item)
                data.score.append(score)


                data.train = torch.cat([data.train,data.imgs_small],0)
                file_name = 'image_' + file.split('-')[0]
                data.filename.append(file_name)


        data.out.append(data.train)
        data.scores.append(data.score)
        data.filenames.append(data.filename)

    data.totalLength = len(data.out)
    # data.totalLength = totalLength
    # find a picture to plot as the experimental reference



    # print("=====================Finished===========================")

    return data

######################################## DATA SET CLASS #########################################

class TurbDataset(Dataset):


    def __init__(self):


        # load & normalize data
        dataDir = ".\inputs\Train Data"
        self = imageLoader_train(self, dataDir)
        self.inputs = self.train
        self.output = self.win
        self.totalLength = self.totalLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.train[idx], self.win[idx]
    #
    # #  reverts normalization
    # def denormalize(self, data):
    #     a = data.copy()
    #     a[0, :, :] /= (1.0/self.max_imgs_0)
    #     # a[1, :, :] /= (1.0/self.max_imgs_1)
    #     # a[2, :, :] /= (1.0/self.max_imgs_2)
    #
    #     return a
# simplified validation data set (main one is TurbDataset above)

class ValiDataset(Dataset):
    def __init__(self):
        dataDir = ".\inputs\Validation Data"
        self = imageLoader_vali(self, dataDir)
        self.totalLength = self.totalLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.out[idx], self.scores[idx], self.filenames[idx], self.foldername[idx]

class TestDataset(Dataset):
    def __init__(self):
        dataDir = ".\inputs\Test Data"
        self = imageLoader_vali(self, dataDir)
        self.totalLength = self.totalLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.out[idx], self.scores[idx], self.filenames[idx], self.foldername[idx]

class CNN_Dataset(Dataset):


    def __init__(self, dataDir):


        # load & normalize data
        #dataDir = "./inputs/Train Data"
        self = imageLoader_train(self, dataDir)
        self.inputs = self.train
        self.output = self.win
        self.totalLength = self.totalLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.train[idx], self.win[idx]