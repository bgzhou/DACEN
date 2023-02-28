#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DACEN.py
@Time    :   2023/02/28 14:19:53
@Author  :   Binggui ZHOU
@Version :   1.0
@Contact :   binggui.zhou[AT]connect.um.edu.mo
@License :   (C)Copyright 2018-2023, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   None
'''
import math
import torch
from torch import nn
from torch.utils.data import Dataset
from einops import rearrange

#=======================================================================================================================
#=======================================================================================================================
# DataLoader Defining

class DatasetFolder(Dataset):
    def __init__(self, matInput, matLabel):
        self.input, self.label = matInput, matLabel
    def __getitem__(self, index):
        return self.input[index], self.label[index]
    def __len__(self):
        return self.input.shape[0]

class DatasetFolder_weights(Dataset):
    def __init__(self, matInput, matLabel, weights):
        self.input, self.label, self.weights = matInput, matLabel, weights
    def __getitem__(self, index):
        return self.input[index], self.label[index], self.weights[index], 
    def __len__(self):
        return self.label.shape[0]

class DatasetFolder_eval(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]

#=======================================================================================================================
#=======================================================================================================================
# Module and Model Defining

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, 'b (rx tx) dmodel -> b dmodel rx tx', rx=4)
        _x = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x) * _x
        x = rearrange(x, 'b dmodel rx tx -> b (rx tx) dmodel')
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class SpatialAttentionModule(nn.Module):

    def __init__(self, d_model, ffn_hidden, drop_prob):
        super(SpatialAttentionModule, self).__init__()
        self.attention = SpatialAttention(kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):

        _x = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x

class DualAttentionChannelEstimationNetwork(nn.Module):
    def __init__(self, input_length):
        super(DualAttentionChannelEstimationNetwork, self).__init__()
        pilot_num = int(input_length / 8)
        
        d_model = 512
        d_hid = 512
        dropout = 0.0
        nlayers = 8
        self.fc1 = nn.Linear(2*pilot_num, d_model)
        self.sa_layers = nn.ModuleList([SpatialAttentionModule(d_model=d_model,
                                                  ffn_hidden=d_hid,
                                                  drop_prob=dropout)
                                     for _ in range(nlayers)])
        self.fc2 = nn.Linear(d_model, 64*2)
        

        d_model = 512
        nhead = 2
        d_hid = 512
        dropout = 0.0
        nlayers = 8
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.ta_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
                                     for _ in range(nlayers)])
        self.fc3 = nn.Linear(4*32, d_model)

        self.fc4 = nn.Linear(d_model, 4*32)

    def forward(self, x):
        out = rearrange(x, 'b c rx (rb subc) sym -> b (rx subc sym) (c rb)', subc=8)
        out = self.fc1(out)
        for layer in self.sa_layers:
            out = layer(out)
        out = self.fc2(out)
        out = rearrange(out, 'b (rx tx) (c d) -> (c d) b (rx tx)', c=2, rx=4)
        out = self.fc3(out)
        out = self.pos_encoder(out)
        for layer in self.ta_layers:
            out = layer(out)
        out = self.fc4(out)
        out = rearrange(out, '(c d) b (rx tx) -> b c rx tx d', c=2, rx=4)
        return out

class DualAttentionChannelEstimationNetwork_low(nn.Module):
    def __init__(self, input_length):
        super(DualAttentionChannelEstimationNetwork_low, self).__init__()
        pilot_num = int(input_length / 8)
        
        d_model = 512
        d_hid = 512
        dropout = 0.0
        nlayers = 8
        self.fc1_low = nn.Linear(2*pilot_num, d_model)
        self.sa_layers = nn.ModuleList([SpatialAttentionModule(d_model=d_model, ffn_hidden=d_hid, drop_prob=dropout)
                                     for _ in range(nlayers)])
        self.fc2 = nn.Linear(d_model, 64*2)
        
        d_model = 512
        nhead = 2
        d_hid = 512
        dropout = 0.0
        nlayers = 8
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.ta_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
                                     for _ in range(nlayers)])
        self.fc3 = nn.Linear(4*32, d_model)

        self.fc4 = nn.Linear(d_model, 4*32)

    def forward(self, x):
        out = rearrange(x, 'b c rx (rb subc) sym -> b (rx subc sym) (c rb)', subc=8)
        out = self.fc1_low(out)
        for layer in self.sa_layers:
            out = layer(out)
        out = self.fc2(out)
        out = rearrange(out, 'b (rx tx) (c d) -> (c d) b (rx tx)', c=2, rx=4)
        out = self.fc3(out)
        out = self.pos_encoder(out)
        for layer in self.ta_layers:
            out = layer(out)
        out = self.fc4(out)
        out = rearrange(out, '(c d) b (rx tx) -> b c rx tx d', c=2, rx=4)
        return out