################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Helper functions for image output
#
############################################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable

import math, re, os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import cm
import matplotlib.pyplot as plt
from enum import Enum
from PIL import Image



####################################### 增加的预定义工具 #################################################

# 算法选择
class Methods(Enum):
    ori_Mode          = 0

    target_Mode       = 11
    surveillance_Mode = 12
    mixTS_Mode        = 13

class Operate():
    conv2d_dx1_center = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
    conv2d_dx1_center.weight.data = torch.Tensor(
        [[[
            [0,0,0],
            [0.5,0,-0.5],
            [0,0,0]
        ]]]
    )
    conv2d_dy1_center = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
    conv2d_dy1_center.weight.data = torch.Tensor(
        [[[
            [0,0.5,0],
            [0,0,0],
            [0,-0.5,0]
        ]]]
    )

    def filter2d_dx1_center(self, input):
        x1, x2 = input.shape[0], input.shape[1]
        temp_tensor = torch.FloatTensor(1, 1, x1, x2)
        temp_tensor[0][0].data.copy_(torch.from_numpy(input))
        output = torch.FloatTensor(x1, x2)
        output[1:(x1 - 1), 1:(x2 - 1)] = self.conv2d_dx1_center(temp_tensor)
        output[0          ] = output[1          ]
        output[x1 - 1     ] = output[x1 - 2     ]
        output[ : , 0     ] = output[ : , 1     ]
        output[ : , x2 - 1] = output[ : , x2 - 2]
        return output.detach().numpy()

    def filter2d_dy1_center(self, input):
        x1, x2 = input.shape[0], input.shape[1]
        temp_tensor = torch.FloatTensor(1, 1, x1, x2)
        temp_tensor[0][0].data.copy_(torch.from_numpy(input))
        output = torch.FloatTensor(x1, x2)
        output[1:(x1 - 1), 1:(x2 - 1)] = self.conv2d_dy1_center(temp_tensor)
        output[0          ] = output[1          ]
        output[x1 - 1     ] = output[x1 - 2     ]
        output[ : , 0     ] = output[ : , 1     ]
        output[ : , x2 - 1] = output[ : , x2 - 2]
        return output.detach().numpy()

##########################################################################################################

# add line to logfiles
def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint: print(line)

# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()

# compute learning rate with decay in second half
def computeLR(i,epochs, minLR, maxLR):
    if i < epochs*0.5:
        return maxLR
    e = (i/float(epochs)-0.5)*2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e*(fmax-fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR-minLR)*f

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()-rect.get_width()/6., 1.03 * height, '{:.3f}' .format(height), size=8)

# image output Old fasion
def image_vali(epoch, i, scores_vali, pred_score, photoname_vali, foldername, corr_pearson, corr_spearman, error_vali):
    # 这两行代码解决 plt 中文显示的问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False


    bar_width = 0.3  # 条形宽度
    index_survey = np.arange(len(scores_vali))  # 男生条形图的横坐标
    index_pred = index_survey + bar_width  # 女生条形图的横坐标

    rect1 = plt.bar(index_survey, height=scores_vali, width=bar_width, color='b', label='survey')
    autolabel(rect1)
    rect2 = plt.bar(index_pred, height=pred_score, width=bar_width, color='g', label='prediction by CNN')
    autolabel(rect2)



    plt.legend()  # 显示图例
    plt.xticks(index_survey + bar_width / 2, photoname_vali)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('Winning probability')  # 纵坐标轴标题
    plt.title(foldername + ': Comparison between survey and prediction\nThe Pearson´s correlation is {:.2}\nThe Spearman´s correlation is {:.2}\nThe relative error is {:.2%}'.format(corr_pearson, corr_spearman, error_vali))  # 图形标题


    #保存图片
    plt.savefig('./result_train/result_validation'+ format(epoch)+'_'+format(i) + '.png')
    plt.cla()
    plt.close()


def image_test(epoch, i, scores_vali, pred_score, photoname_vali, foldername, corr_pearson, corr_spearman, error_test):
    # 这两行代码解决 plt 中文显示的问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    bar_width = 0.3  # 条形宽度
    index_survey = np.arange(len(scores_vali))  # 男生条形图的横坐标
    index_pred = index_survey + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    rect1 = plt.bar(index_survey, height=scores_vali, width=bar_width, color='b', label='survey')
    autolabel(rect1)
    rect2 = plt.bar(index_pred, height=pred_score, width=bar_width, color='g', label='prediction by CNN')
    autolabel(rect2)

    plt.legend()  # 显示图例
    plt.xticks(index_survey + bar_width / 2,
               photoname_vali)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('Winning probability')  # 纵坐标轴标题
    plt.title(foldername + ': Comparison between survey and prediction\nThe Pearson´s correlation is {:.2}\nThe Spearman´s correlation is {:.2}\n The relative error is {:.2%}'.format(corr_pearson, corr_spearman, error_test))  # 图形标题


    # 保存图片
    plt.savefig('./result_train/result_test' + format(epoch) + '_' + format(i) + '.png')
    plt.cla()
    plt.close()

# read data split from command line
def readProportions():
    flag = True
    while flag:
        input_proportions = input("Enter total numer for training files and proportions for training (normal, superimposed, sheared respectively) seperated by a comma such that they add up to 1: ")
        input_p = input_proportions.split(",")
        prop = [ float(x) for x in input_p ]
        if prop[1] + prop[2] + prop[3] == 1:
            flag = False
        else:
            print("\n\nError: poportions don't sum to 1")
            print("##################################\n")
    return(prop)

# helper from data/utils
def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


