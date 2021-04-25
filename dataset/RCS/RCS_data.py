# -*- coding: utf-8 -*-

import torch
import pandas as pd
import os
import numpy as np
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def transfer_format_3x3(input_array):
    x0 = input_array[0]
    x1 = input_array[1]
    x2 = input_array[2]
    x3 = input_array[3]
    x4 = input_array[4]
    x5 = input_array[5]
    x6 = input_array[6]
    x7 = input_array[7]
    x8 = input_array[8]
      
    data = [x0,x1,x2,x3,x4,x5,x6,x7,x8]
    result = np.array(data).reshape(3,3)
    return result

class RCS_data_from_excel():
    def __init__(self, csv_file):
        self.read = pd.read_excel(csv_file)
        read_data = self.read.iloc[1: ,: ].values
        get_data = []
        for i in range(read_data.shape[1]):
            data = transfer_format_3x3(self.read.iloc[1: , i].values)
            get_data.append(data)
        data_x = np.array(get_data)
        self.data_x_vector = data_x.reshape(len(data_x),3,3,1).astype('float32') # (75, 3, 3, 1)
        
        zeros = np.zeros((read_data.shape[1]), dtype=int )
        for i in range(len(zeros)):
            step = int(len(zeros)/5)
            if i < step:
                zeros[i] = 1
            elif step <= i < 2*step:
                zeros[i] = 2
            elif 2*step <= i < 3*step:
                zeros[i] = 3
            elif 3*step <= i < 4*step:
                zeros[i] = 4
            else:
                zeros[i] = 5
        
        # Convert to torch Tensor
        self.label = torch.as_tensor(zeros)
        #self.label = torch.nn.functional.one_hot(labels_tensor.to(torch.int64), num_classes=5)
        #print(self.data_label_onehot)
        
    def __len__(self):
        return self.read.shape[1]

    def __getitem__(self, idx):
        img = self.data_x_vector[idx]
        return img,self.label[idx]
        

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    batch_size = 5
    dataset = RCS_data_from_excel("train_angle_5_ship.xlsx")
    print(len(dataset))
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # TODO

    
