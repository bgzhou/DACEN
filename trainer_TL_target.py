#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trainer_TL_target.py
@Time    :   2023/02/28 15:53:49
@Author  :   Binggui ZHOU
@Version :   1.0
@Contact :   binggui.zhou[AT]connect.um.edu.mo
@License :   (C)Copyright 2018-2023, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   None
'''

import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from DACEN import *
from utils import *
#=======================================================================================================================
#=======================================================================================================================
# Training Configurations

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_RX = 4
NUM_TX = 32
BATCH_SIZE = 256
EPOCHS = 1000
LEARNING_RATE = 6e-5
NUM_PILOT = 2
NUM_SUBC = NUM_PILOT * 8

training_index = 'DEFINE_YOUR_TRAINING_INDEX'
save_dir = './output/%s/'%(training_index)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#=======================================================================================================================
#=======================================================================================================================
# Data Loading

# load pilot
pilot = np.load("PATH_TO_THE_26RB_PILOTS")
pilot = rearrange(pilot, 'b c rx (rb subc) sym -> b (c rx) (subc sym) rb', subc=8)

# load ht
ht = np.load("PATH_TO_THE_CHANNEL_LABEL")

# split data
split_valid_idx, split_test_idx = int(0.7 * pilot.shape[0]), int(0.8 * pilot.shape[0])
pilot_train, pilot_valid, pilot_test = pilot[:split_valid_idx,...], pilot[split_valid_idx:split_test_idx,...],  pilot[split_test_idx:,...]
ht_train, ht_valid, ht_test = ht[:split_valid_idx,...], ht[split_valid_idx:split_test_idx,...],  ht[split_test_idx:,...]

# process training data (pilot, channel, and cos similarity)
index_list = [[0,4],[2,6]]
train_pilots = []
train_labels = []
instance_weights = []

num_total_samples = 0
_total_samples = []
_total_sample_weights = []
_total_sample_labels = []
for indices in index_list:
    _pilot_train = pilot_train[..., indices]
    _pilot_train = rearrange(_pilot_train, 'b (c rx) (subc sym) rb -> b (subc sym) (c rx rb)', c=2, subc=8)
    ref_pilot_train = pilot_train[..., [1,5]]
    ref_pilot_train = rearrange(ref_pilot_train, 'b (c rx) (subc sym) rb -> b (subc sym) (c rx rb)', c=2, subc=8)
    img_total = 2 * 4 * len(index_list[0])
    num_sample_subband = 8 * 4
    for j, (_pilot, _ref_pilot) in enumerate(zip(_pilot_train, ref_pilot_train)):
        _ref_pilot2 = _ref_pilot[0:num_sample_subband, 0:int(img_total):2] + 1j*_ref_pilot[0:num_sample_subband, 1:int(img_total):2]
        _pilot2 = _pilot[0:num_sample_subband, 0:int(img_total):2] + 1j*_pilot[0:num_sample_subband, 1:int(img_total):2]
        score_cos = 0
        for i in range(num_sample_subband):
            _ref_pilot2_sample = _ref_pilot2[i:i+1,]
            _pilot2_sample = _pilot2[i:i+1,]
            score_tmp = cos_sim(_ref_pilot2_sample,_pilot2_sample)
            score_cos = score_cos + abs(score_tmp)*abs(score_tmp)
        score_cos = score_cos/num_sample_subband
        if score_cos >= 0.9:
            num_total_samples += 1
            _total_samples.append(rearrange(_pilot, '(subc sym) (c rx rb) -> 1 c rx (rb subc) sym', c=2, rx =4, subc=8))
            _total_sample_weights.append(score_cos)
            _total_sample_labels.append(ht_train[j:j+1,...])
    train_pilots.append(np.concatenate(_total_samples))
    train_labels.append(np.concatenate(_total_sample_labels))
    instance_weights.append(np.array(_total_sample_weights))

index_list = [[1,5]]
for indices in index_list:
    _pilot_train = pilot_train[..., indices]
    _pilot_train = rearrange(_pilot_train, 'b (c rx) (subc sym) rb -> b c rx (rb subc) sym', c=2, subc=8)
    train_pilots.append(_pilot_train)
    train_labels.append(ht_train)
    instance_weights.append(np.ones(ht_train.shape[0]))

pilot_train = np.concatenate(train_pilots)
ht_train = np.concatenate(train_labels)
instance_weights = np.concatenate(instance_weights)
train_dataset = DatasetFolder_weights(pilot_train, ht_train, instance_weights)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)

# process validation data (pilot, channel)
index_list = [[1,5]]
valid_pilots = []
valid_labels = []
for indices in index_list:
    _pilot_valid = pilot_valid[..., indices]
    _pilot_valid = rearrange(_pilot_valid, 'b (c rx) (subc sym) rb -> b c rx (rb subc) sym', c=2, subc=8)
    valid_pilots.append(_pilot_valid)
    valid_labels.append(ht_valid)

pilot_valid = np.concatenate(valid_pilots)
ht_valid = np.concatenate(valid_labels)
valid_dataset = DatasetFolder(pilot_valid, ht_valid)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

len_train = len(ht_train)
len_valid = len(ht_valid)

del pilot, pilot_train, pilot_valid, pilot_test, ht, ht_train, ht_valid, ht_test, train_dataset, valid_dataset
gc.collect()

print('data loading is finished ...')

#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving

model_ce = DualAttentionChannelEstimationNetwork_low(NUM_SUBC).cuda()
model_ce.load_state_dict(torch.load("PATH_TO_YOUR_TRAINED_SOURCE_MODEL")['state_dict'], strict=False)

criterion = nn.MSELoss().cuda()

def weighted_loss(y, y_hat, w):
    return (criterion(y, y_hat)*w).mean()

optimizer = torch.optim.Adam(model_ce.parameters(), lr=LEARNING_RATE)

bestLoss = 10
for epoch in range(EPOCHS):
    start = time.time()
    model_ce.train()
    trainLoss = 0
    for i, (modelInput, label, weights) in enumerate(train_loader):
        modelInput, label, weights = modelInput.cuda(), label.cuda(), weights.cuda()
        modelOutput = model_ce(modelInput)
        loss = weighted_loss(label, modelOutput, weights)
        trainLoss += loss.item() * modelInput.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avgTrainLoss = trainLoss / len_train
    model_ce.eval()
    testLoss = 0
    with torch.no_grad():
        for i, (modelInput, label) in enumerate(valid_loader):
            modelInput, label = modelInput.cuda(), label.cuda()
            modelOutput = model_ce(modelInput)
            testLoss += criterion(label, modelOutput).item() * modelInput.size(0)
        avgTestLoss = testLoss / len_valid
        print('Epoch:[{0}]\t' 'Train Loss:{loss1:.5f}\t' 'Val Loss:{loss2:.5f}\t' 'Time:{time:.1f}secs\t'.format(epoch, loss1=avgTrainLoss, loss2=avgTestLoss, time=time.time()-start))
        if avgTestLoss < bestLoss:
            bestLoss = avgTestLoss
            torch.save({'state_dict': model_ce.state_dict(), }, os.path.join(save_dir, 'TF_target_sim%.4f_ep%s.pth.tar'%(bestLoss,epoch)))
            print("Model saved")
print('Training is finished!')