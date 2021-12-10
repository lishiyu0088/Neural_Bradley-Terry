from torch.utils.data import DataLoader
from dataset import CNN_Dataset
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
import argparse

######################################################### 1-read data  #########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="please assign model type", default="vgg")
args = parser.parse_args()
CNN_model = args.model


print(os.getcwd())


runningMode = Methods.target_Mode
prop=None
batch_size = 1
dir = './inputs/Train Data/'



targets_dn = torch.FloatTensor(10, 3, 128, 128)


targets_dn = Variable(targets_dn)
targets_dn = targets_dn.cuda()

outputs_dn = torch.FloatTensor(10, 3, 128, 128)
outputs_dn = Variable(outputs_dn)
outputs_dn = outputs_dn.cuda()


data = CNN_Dataset(dir)
length = data.totalLength
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)







########################################################## 2-Train mode #########################################################
expo = 5
dropout = 0.

netD = DistanceModel(baseType=CNN_model,useGPU=True)
netD.cuda()
netD.load_state_dict(torch.load("./result_train/"+CNN_model+"/modelDtarget_Mode"))

####################### 2-1 Define error function

criterionL1 = nn.L1Loss()
criterionL1.cuda()


####################### 2-2 Define optimizer
lrG = 0.00005
decayLr = True

# Use Adam
optimizerD = optim.Adam(netD.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)



######################################### Train process ##############################################################

epochs = 30
epoch_train = []
epoch_vali = []
loss_train = []
corr_vali_plt = []


for epoch in range(epochs):
    print("Starting epoch {} / {} \n".format((epoch+1),epochs))

    netD.train()

    loss_accu = 0.0
    for traindata in trainLoader:
        inputs_cpu, targets_cpu = traindata

            
        inputs_cpu, targets_cpu  = inputs_cpu.float(),targets_cpu.float()
        targets_gpu = targets_cpu.cuda()
        input2, input3 = inputs_cpu.split([3, 3], dim=1)
        input2 = input2.cuda()
        input3 = input3.cuda()

        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
            if currLr < lrG:
                for g in optimizerD.param_groups:
                    g['lr'] = currLr


        netD.zero_grad()
        d1 = netD(input2)
        d2 = netD(input3)
        prediction = torch.abs(d1)*torch.reciprocal_(torch.abs(d1)+torch.abs(d2))
        lossL1 = torch.mean(torch.abs(prediction-targets_gpu))
        loss_accu += float(lossL1)

        lossL1.backward()

        optimizerD.step()
    loss = loss_accu*batch_size/length

    epoch_train.append(epoch+1)
    loss_train.append(100*loss)

    print('The average loss is {:.3%}'.format(loss))


     
    torch.save(netD.state_dict(), "./result_train/modelD" + runningMode.name)

########################################################## 3-Validation mode #########################################################
    if (epoch+1)%5 == 0:

        netD.load_state_dict(torch.load("./result_train/modelD" + runningMode.name))

        criterionL1 = nn.L1Loss()

        netD.eval()



        Validata = ValiDataset()
        corr_vali_sum_pearson = 0
        corr_vali_sum_spearman = 0
        error_vali_sum =0
        scores_vali_all = np.empty(0)
        pred_vali_all = np.empty(0)

        for i, validata in enumerate(Validata):

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
        epoch_vali.append(epoch+1)
        corr_vali_plt.append(corr_vali_ave_spearman)
        error_vali_ave = error_vali_sum / (i+1)
        print('Validation: The average Pearson´s covariance is {:.3}'.format(corr_vali_ave_pearson))  # 输出每次的误差值
        print('Validation: The average Spearman´s covariance is {:.3}'.format(corr_vali_ave_spearman))  # 输出每次的误差值
        print('Validation: The average relative error is {:.3%}'.format(error_vali_ave))
        plt.scatter(100*scores_vali_all, 100*pred_vali_all)
        plt.title('Scatter plot:\nThe average Pearson´s covariance is {:.3}\nThe average Spearman´s covariance is {:.3}\nThe average relative error is {:.3%}'.format(corr_vali_ave_pearson,corr_vali_ave_spearman, error_vali_ave))  # 图形标题
        plt.xlabel('Winning probability of survey in %')  # 纵坐标轴标题
        plt.ylabel('Winning probability of prediction in %')  # 纵坐标轴标题
        plt.savefig('./result_train/result_vali_scatter' + format(epoch) + '.png')
        plt.cla()
        plt.close()
########################################################## 4-Test mode #########################################################

netD.load_state_dict(torch.load("./result_train/modelD" + runningMode.name))

criterionL1 = nn.L1Loss()
#criterionL1.cuda()

netD.eval()



Testdata = TestDataset()
corr_test_sum_pearson = 0
corr_test_sum_spearman = 0
error_test_sum =0
scores_test_all = np.empty(0)
pred_test_all = np.empty(0)

for i, testdata in enumerate(Testdata):
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
print('Test: The average Pearson´s covariance is {:.3%}'.format(corr_test_ave_pearson))
print('Test: The average Spearman´s covariance is {:.3%}'.format(corr_test_ave_spearman))
print('Test: The average relative error is {:.3%}'.format(error_vali_ave))
plt.scatter(100 * scores_test_all, 100 * pred_test_all)
plt.title('Scatter plot:\nThe average Pearson´s covariance is {:.3}\nThe average Spearman´s covariance is {:.3}\nThe average relative error is {:.3%}'.format(
    corr_test_ave_pearson, corr_test_ave_spearman, error_test_ave))
plt.xlabel('Winning probability of survey in %')
plt.ylabel('Winning probability of prediction in %')
plt.savefig('./result_train/result_test_scatter' + format(epoch) + '.png')
plt.cla()
plt.close()

plt.plot(epoch_train, loss_train)
plt.title('Training loss in epochs')
plt.xlabel('Epoch')
plt.ylabel('Training loss in %')
plt.savefig('./result_train/training_loss.png')
plt.cla()
plt.close()

plt.plot(epoch_vali, corr_vali_plt)
plt.title('Validation correlation in epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation correlation')
plt.savefig('./result_train/corr_vali_plt.png')
plt.cla()
plt.close()