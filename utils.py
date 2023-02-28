#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/02/28 15:21:32
@Author  :   Binggui ZHOU
@Version :   1.0
@Contact :   binggui.zhou[AT]connect.um.edu.mo
@License :   (C)Copyright 2018-2023, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   None
'''

import numpy as np

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1*num2))
    return cos.item()