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




class Methods(Enum):
    ori_Mode          = 0

    target_Mode       = 11
    surveillance_Mode = 12
    mixTS_Mode        = 13


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

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False


    bar_width = 0.3
    index_survey = np.arange(len(scores_vali))
    index_pred = index_survey + bar_width

    rect1 = plt.bar(index_survey, height=scores_vali, width=bar_width, color='b', label='survey')
    autolabel(rect1)
    rect2 = plt.bar(index_pred, height=pred_score, width=bar_width, color='g', label='prediction by CNN')
    autolabel(rect2)



    plt.legend()
    plt.xticks(index_survey + bar_width / 2, photoname_vali) 
    plt.ylabel('Winning probability')
    plt.title(foldername + ': Comparison between survey and prediction\nThe Pearson´s correlation is {:.2}\nThe Spearman´s correlation is {:.2}\nThe relative error is {:.2%}'.format(corr_pearson, corr_spearman, error_vali))



    plt.savefig('./result_train/result_validation'+ format(epoch)+'_'+format(i) + '.png')
    plt.cla()
    plt.close()


def image_test(epoch, i, scores_vali, pred_score, photoname_vali, foldername, corr_pearson, corr_spearman, error_test):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    bar_width = 0.3 
    index_survey = np.arange(len(scores_vali)) 
    index_pred = index_survey + bar_width


    rect1 = plt.bar(index_survey, height=scores_vali, width=bar_width, color='b', label='survey')
    autolabel(rect1)
    rect2 = plt.bar(index_pred, height=pred_score, width=bar_width, color='g', label='prediction by CNN')
    autolabel(rect2)

    plt.legend() 
    plt.xticks(index_survey + bar_width / 2,
               photoname_vali) 
    plt.ylabel('Winning probability') 
    plt.title(foldername + ': Comparison between survey and prediction\nThe Pearson´s correlation is {:.2}\nThe Spearman´s correlation is {:.2}\n The relative error is {:.2%}'.format(corr_pearson, corr_spearman, error_test))


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


