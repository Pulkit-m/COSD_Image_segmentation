import torch 
import os
import torch.nn as nn
import torch.optim as optim
import math
import os
import random
import numpy as np
import pandas as pd
from unet.unet_model import UNet
from DataLoader import COSD_Dataset_Loader, ToTensor, Generate_csv_logs_for_images
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib
import matplotlib.pyplot as plt 


def dataset_train_test_split(dataset,split_ratio=0.15):
    """
        Returns a dictionary with keys 'train', 'val'
        returned objects can be unpacked and directly passed into torch.utils.data.DataLoader() function
    """
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=split_ratio)
    # print(train_idx)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def visualizeData(dataset, index = 1, num = 3 ,show = False):
    plt.figure(figsize=(10,4.8))
    idx = index
    order = ['rgb_image','depth_image','mask_image']    

 
    for i in range(3):
        ax = plt.subplot(1,3,i+1)
        ax.set_title(f'sample #{i}')
        image = np.array(dataset[idx][order[i]])
        image = image.transpose((1,2,0))
        plt.imshow(image)

    if show: plt.show()
    if os.path.exists('plots'):
        plt.savefig(f'plots/dataset_sample')
    

def main():
    #Generating a csv file for dataset
    if os.path.isfile('train/csv_file.csv') == False:
        Generate_csv_logs_for_images(column_names= ['rgb_names', 'depth_names', 'mask_names', 'Mirror','Transparent']
                                ,rgb_path = 'train/rgb'
                                ,depth_path = 'train/depth'
                                ,mask_path = 'train/mask'
                                ,mirror_path = 'train/train_mirror.txt'
                                ,transparent_path= 'train/train_trans.txt'
                                )

    #loading dataset
    dataset_input = COSD_Dataset_Loader(rgb_dir_path= 'train/rgb'
                            , depth_dir_path= 'train/depth'
                            , csv_file_path= 'train/csv_file.csv'
                            , mask_dir_path= 'train/mask'
                            , data_type= 'i'
                            , transform = transforms.Compose([ToTensor()]))

    
    # performing train test split
    datasets = dataset_train_test_split(dataset_input)
    trainDataset, testDataset = datasets['train'], datasets['val']

    #visualizing data. look png file in plots/dataset_sample.png
    visualizeData(dataset= trainDataset, show=False)


    # num_epochs = 5

    # for epoch in num_epochs:

    #performing a sample forward pass and printing the output:

    trainData = torch.utils.data.DataLoader(dataset = trainDataset, batch_size= 3, shuffle= False)
    network = UNet()
    for batch_i, batch in enumerate(trainData):
        rgb_batch = batch['rgb_image']/255
        print(rgb_batch.size())
        depth_batch = batch['depth_image']/255
        print(depth_batch.size())
        mask_batch = batch['mask_image']

        logits = network.forward(rgb_batch, depth_batch)
        print(logits.size())
        # print(logits[0])
        loss = nn.CrossEntropyLoss()
        output = loss(logits, mask_batch)
        print(output)
        break










if __name__ == '__main__': main()

