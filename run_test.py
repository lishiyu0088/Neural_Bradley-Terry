
import dataset
from torch.utils.data import DataLoader
from dataset import TurbDataset
from dataset import ValiDataset
from dataset import TestDataset
from utils import Methods
from DfpNet import TurbNetD
from DfpNet import DistanceModel
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import os
from torch.nn import functional as F
import math

runningMode = Methods.target_Mode  #methods从utils里找，target_Mode = 11
prop=None  #by default, use all from "../data/train"。 定义prop，之后用
batch_size = 1  #定义读取数据大小，之后用。每次读取10张图
dir = './inputs/Train Data/'  #定义路径，之后用
netD = DistanceModel()
netD.cuda()
epoch=49

########################################################## 3-Validation模式，使用训练好的参数出图 #########################################################

netD.load_state_dict(torch.load("./result_train/old/lsim/modelD" + runningMode.name))

criterionL1 = nn.L1Loss()
#criterionL1.cuda()

netD.eval()



Validata = ValiDataset()
corr_vali_sum_pearson = 0
corr_vali_sum_spearman = 0
error_vali_sum =0
scores_vali_all = np.empty(0)
pred_vali_all = np.empty(0)

for i, validata in enumerate(Validata):   # 读入训练好的数据

    pred_score_vali = []
    pred_sum_vali = 0.0
    survey_sum = 0.0

    inputs_vali, scores_vali, photoname_vali, foldername_vali = validata
    for input_vali, score_vali in zip(inputs_vali, scores_vali):

        input = torch.unsqueeze(input_vali, 0)
        input = input.cuda()
        score_exp = float(abs(netD(input)))
        pred_score_vali.append(score_exp)
        pred_sum_vali = pred_sum_vali + score_exp
        survey_sum = survey_sum + score_vali


    #还在for循环之内，使用训练好的参数出图并保存
    scores_vali = np.array(scores_vali)
    pred_score_vali = np.array(pred_score_vali)

    pred_score_vali = pred_score_vali / pred_sum_vali
    pred_vali_all = np.hstack((pred_vali_all, pred_score_vali))
    scores_vali = scores_vali / survey_sum
    scores_vali_all = np.hstack((scores_vali_all, scores_vali))

    corr_vali_pearson, _ = pearsonr(scores_vali, pred_score_vali)
    corr_vali_sum_pearson = corr_vali_sum_pearson + corr_vali_pearson

    corr_vali_spearman, _ = spearmanr(scores_vali, pred_score_vali)
    corr_vali_sum_spearman = corr_vali_sum_spearman + corr_vali_spearman

    error_vali_list = np.abs(scores_vali - pred_score_vali) / pred_score_vali
    error_vali = np.mean(error_vali_list)
    error_vali_sum = error_vali_sum + error_vali
    utils.image_vali(epoch, i, scores_vali, pred_score_vali, photoname_vali, foldername_vali, corr_vali_pearson, corr_vali_spearman, error_vali)
corr_vali_ave_pearson = corr_vali_sum_pearson / (i+1)
corr_vali_ave_spearman = corr_vali_sum_spearman / (i + 1)
error_vali_ave = error_vali_sum / (i+1)
print('Validation: The average Pearson´s covariance is {:.2}'.format(corr_vali_ave_pearson))  # 输出每次的误差值
print('Validation: The average Spearman´s covariance is {:.2}'.format(corr_vali_ave_spearman))  # 输出每次的误差值
print('Validation: The average relative error is {:.2%}'.format(error_vali_ave))
plt.scatter(100*scores_vali_all, 100*pred_vali_all)
plt.title('Scatter plot:\nThe average Pearson´s covariance is {:.2}\nThe average Spearman´s covariance is {:.2}\nThe average relative error is {:.2%}'.format(corr_vali_ave_pearson,corr_vali_ave_spearman, error_vali_ave))  # 图形标题
plt.xlabel('Winning probability of survey in %')  # 纵坐标轴标题
plt.ylabel('Winning probability of prediction in %')  # 纵坐标轴标题
plt.savefig('./result_train/result_vali_scatter' + format(epoch) + '.png')
plt.cla()
plt.close()
########################################################## 4-Test模式，使用训练好的参数出图 #########################################################
criterionL1 = nn.L1Loss()
#criterionL1.cuda()

netD.eval()



Testdata = TestDataset()
corr_test_sum_pearson = 0
corr_test_sum_spearman = 0
error_test_sum =0
scores_test_all = np.empty(0)
pred_test_all = np.empty(0)

for i, testdata in enumerate(Testdata):   # 读入训练好的数据
    pred_score_test = []
    pred_sum_test = 0.0
    survey_sum = 0.0

    inputs_test, scores_test, photoname_test, foldername_test = testdata
    for input_test, score_test in zip(inputs_test, scores_test):
        input = torch.unsqueeze(input_test, 0)
        input = input.cuda()
        score_exp = float(abs(netD(input)))
        pred_score_test.append(score_exp)
        pred_sum_test = pred_sum_test + score_exp
        survey_sum = survey_sum + score_test


    #还在for循环之内，使用训练好的参数出图并保存
    scores_test = np.array(scores_test)
    pred_score_test = np.array(pred_score_test)


    pred_score_test = pred_score_test / pred_sum_test
    pred_test_all = np.hstack((pred_test_all, pred_score_test))
    scores_test = scores_test / survey_sum
    scores_test_all = np.hstack((scores_test_all, scores_test))
    corr_test_pearson, _ = pearsonr(scores_test, pred_score_test)
    corr_test_sum_pearson = corr_test_sum_pearson + corr_test_pearson

    corr_test_spearman, _ = spearmanr(scores_test, pred_score_test)
    corr_test_sum_spearman = corr_test_sum_spearman + corr_test_spearman
    error_test = np.mean(np.abs(scores_test - pred_score_test) / pred_score_test)
    error_test_sum = error_test_sum + error_test
    utils.image_test(epoch, i, scores_test, pred_score_test, photoname_test, foldername_test, corr_test_pearson, corr_test_spearman, error_test)
corr_test_ave_pearson = corr_test_sum_pearson / (i + 1)
corr_test_ave_spearman = corr_test_sum_spearman / (i + 1)
error_test_ave = error_test_sum / (i+1)
print('Test: The average Pearson´s covariance is {:.2%}'.format(corr_test_ave_pearson))  # 输出每次的误差值
print('Test: The average Spearman´s covariance is {:.2%}'.format(corr_test_ave_spearman))  # 输出每次的误差值
print('Test: The average relative error is {:.2%}'.format(error_test_ave))
plt.scatter(100 * scores_test_all, 100 * pred_test_all)
plt.title('Scatter plot:\nThe average Pearson´s covariance is {:.2}\nThe average Spearman´s covariance is {:.2}\nThe average relative error is {:.2%}'.format(
    corr_test_ave_pearson, corr_test_ave_spearman, error_test_ave))  # 图形标题
plt.xlabel('Winning probability of survey in %')  # 纵坐标轴标题
plt.ylabel('Winning probability of prediction in %')  # 纵坐标轴标题
plt.savefig('./result_train/result_test_scatter' + format(epoch) + '.png')
plt.cla()
plt.close()

