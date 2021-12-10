import dataset
from torch.utils.data import DataLoader
from dataset import imageLoader_AVA
from utils import Methods
from DfpNet import DistanceModel
import torch.nn as nn
import torch.optim as optim
import torch, gc
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import os
from NIMA.nima.common import Transform
from NIMA.nima.dataset import AVADataset
import argparse

######################################################### 1-read data  #########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="please assign model type", default="vgg")
args = parser.parse_args()
CNN_model = args.model
print(os.getcwd())  #print the path of current folder

path_to_save_csv="./images"
path_to_images="./AVA_dataset"
imageLoader_AVA(path_to_images)
transform = Transform()
train_ds = AVADataset(path_to_save_csv + "/" + "train.csv", path_to_images, transform.train_transform)
validation_ds = AVADataset(path_to_save_csv + "/" + "val.csv", path_to_images, transform.train_transform)
test_ds = AVADataset(path_to_save_csv + "/" + "test.csv", path_to_images, transform.train_transform)
trainLoader = DataLoader(train_ds, batch_size=3, shuffle=True)
length = len(trainLoader)
validationLoader = DataLoader(validation_ds, batch_size=3, shuffle=False)
testLoader = DataLoader(test_ds, batch_size=3, shuffle=False)
runningMode = Methods.target_Mode
prop=None  #by default, use all from "../data/train".
batch_size = 3  #Define batch size
dir = './inputs/Train Data/'  #Define path of train data




targets_dn = torch.FloatTensor(10, 3, 128, 128)
targets_dn = Variable(targets_dn)
targets_dn = targets_dn.cuda()   # with cuda

outputs_dn = torch.FloatTensor(10, 3, 128, 128)
outputs_dn = Variable(outputs_dn)
outputs_dn = outputs_dn.cuda()




########################################################## 2-Train mode #########################################################
expo = 5
dropout = 0.

netD = DistanceModel(baseType=CNN_model,useGPU=True)
netD.cuda()


####################### 2-1 Define error function

criterionL1 = nn.L1Loss(reduction=sum, size_average=False)
criterionL1.cuda()


####################### 2-2 Define optimizer
lrG = 0.00005
decayLr = True

# Use Adam optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)



############################################ Train process ###########################################################

epochs = 20 #20 epochs for training
epoch_train = []
epoch_vali = []
loss_train = []
loss_vali = []
corr_vali_plt = []


for epoch in range(epochs):
    print("Starting epoch {} / {} \n".format((epoch+1),epochs))

    netD.train()  # train mode

    loss_accu = 0.0
    for traindata in trainLoader:   #read data
        inputs_cpu, targets_cpu = traindata    #devide input into images and scores

            
        inputs_cpu, targets_cpu  = inputs_cpu.float(),targets_cpu.float()      #change data into float
        inputs_gpu = inputs_cpu.cuda()
        targets_gpu = targets_cpu.cuda()
        targets_gpu = torch.unsqueeze(targets_gpu, dim=1)
        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
            if currLr < lrG:
                for g in optimizerD.param_groups:
                    g['lr'] = currLr


        netD.zero_grad()
        prediction = netD(inputs_gpu)
        lossL1 = criterionL1(prediction,targets_gpu)
        loss_accu += lossL1.item()

        lossL1.backward()

        optimizerD.step()     # adjust step size
    loss = loss_accu/length

    epoch_train.append(epoch+1)
    loss_train.append(loss)

    print('The average loss is {:.2}'.format(loss))  # print error in each epoch


     
    torch.save(netD.state_dict(), "./result_train/modelD" + runningMode.name)   # save trained models

    del lossL1, loss, inputs_cpu, inputs_gpu, targets_cpu, targets_gpu


########################################################## 3-Validation mode #########################################################
    if (epoch+1)%10 == 0:

        netD.load_state_dict(torch.load("./result_train/modelD" + runningMode.name))

        criterionL1 = nn.L1Loss()
        #criterionL1.cuda()

        netD.eval()



        corr_vali_sum_pearson = 0
        corr_vali_sum_spearman = 0
        error_vali_sum = 0
        scores_vali_all = []
        pred_vali_all = []
        loss_vali_sum = 0

        for i, validata in enumerate(validationLoader):
            input_vali_cpu, score_vali_cpu = validata
            input_vali = input_vali_cpu.cuda()
            score_vali = score_vali_cpu.cuda()
            score_vali = torch.unsqueeze(score_vali, dim=1)
            score_vali_list = score_vali_cpu.detach().numpy().tolist()
            scores_vali_all = scores_vali_all + score_vali_list
            pred_vali = netD(input_vali)
            pred_vali_list = pred_vali.cpu().detach()[:,0].numpy().tolist()
            pred_vali_all = pred_vali_all + pred_vali_list
            loss_vali_sum += criterionL1(score_vali, pred_vali).item()

        pred_vali_all_np = np.array(pred_vali_all)
        scores_vali_all_np = np.array(scores_vali_all)
        corr_vali_ave_pearson, _ = pearsonr(pred_vali_all_np, scores_vali_all_np)
        corr_vali_ave_spearman, _ = spearmanr(pred_vali_all_np, scores_vali_all_np)
        error_vali_ave= np.mean(np.abs(pred_vali_all_np - scores_vali_all_np) / scores_vali_all_np)
        loss_vali_ave = loss_vali_sum/(i+1)

        print('Validation: The average Pearson´s covariance is {:.2}'.format(corr_vali_ave_pearson))
        print('Validation: The average Spearman´s covariance is {:.2}'.format(corr_vali_ave_spearman))
        print('Validation: The average relative error is {:.2%}'.format(error_vali_ave))
        plt.scatter(scores_vali_all, pred_vali_all)
        plt.title('Scatter plot of validation:\nThe average Pearson´s covariance is {:.2}\nThe average Spearman´s covariance is {:.2}\nThe average relative error is {:.2%}'.format(corr_vali_ave_pearson,corr_vali_ave_spearman, error_vali_ave))  # 图形标题
        plt.xlabel('Scores of survey in %')
        plt.ylabel('Scores of prediction in %')
        plt.savefig('./result_train/result_vali_scatter' + format(epoch) + '.png')
        plt.cla()
        plt.close()

        epoch_vali.append(epoch + 1)
        loss_vali.append(loss_vali_ave)

########################################################## 4-Test mode #########################################################

netD.load_state_dict(torch.load("./result_train/modelD" + runningMode.name))

criterionL1 = nn.L1Loss()
#criterionL1.cuda()

netD.eval()

corr_test_sum_pearson = 0
corr_test_sum_spearman = 0
error_test_sum = 0
scores_test_all = []
pred_test_all = []

for i, testdata in enumerate(testLoader):
    input_test_cpu, score_test_cpu = testdata
    input_test = input_test_cpu.cuda()
    score_test = score_test_cpu.cuda()
    score_test = torch.unsqueeze(score_test, dim=1)
    score_test_list = score_test_cpu.detach().numpy().tolist()
    scores_test_all = scores_test_all + score_test_list
    pred_test = netD(input_test)
    pred_test_list = pred_test.cpu().detach()[:, 0].numpy().tolist()
    pred_test_all = pred_test_all + pred_test_list

pred_test_all_np= np.array(pred_test_all)
scores_test_all_np = np.array(scores_test_all)
corr_test_ave_pearson, _ = pearsonr(pred_test_all_np, scores_test_all_np)
corr_test_ave_spearman, _ = spearmanr(pred_test_all_np, scores_test_all_np)
error_test_ave = np.mean(np.abs(pred_test_all_np - scores_test_all_np) / scores_test_all_np)

print('Test: The average Pearson´s covariance is {:.2}'.format(corr_test_ave_pearson))
print('Test: The average Spearman´s covariance is {:.2}'.format(corr_test_ave_spearman))
print('Test: The average relative error is {:.2%}'.format(error_test_ave))
plt.scatter(scores_test_all, pred_test_all)
plt.title(
    'Scatter plot of test:\nThe average Pearson´s covariance is {:.2}\nThe average Spearman´s covariance is {:.2}\nThe average relative error is {:.2%}'.format(
        corr_test_ave_pearson, corr_test_ave_spearman, error_test_ave))
plt.xlabel('Scores of survey')
plt.ylabel('Scores of prediction')
plt.savefig('./result_train/result_test_scatter' + format(epoch) + '.png')
plt.cla()
plt.close()

plt.plot(epoch_train, loss_train)
plt.title('Training loss in epochs')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.savefig('./result_train/training_loss.png')
plt.cla()
plt.close()

plt.plot(epoch_vali, loss_vali)
plt.title('Validation loss in epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation loss')
plt.savefig('./result_train/corr_vali_plt.png')
plt.cla()
plt.close()