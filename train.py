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



def IoU(predicted_batch, mask_batch):
    IoU_ = 0
    intersection = np.logical_and(predicted_batch, mask_batch)
    union = np.logical_or(predicted_batch, mask_batch)
    try:
        np.sum(intersection)/np.sum(union)
    except ZeroDivisionError:
        IoU = 0
    return IoU_

def rmse(predicted_batch, mask_batch):
    return np.sqrt(np.mean(np.square(mask_batch - predicted_batch)))

def accuracy(predicted_batch, mask_batch):
    pass


# def train_model(neuralnet, train_dataset, validation_dataset, optimizer= 'Adam', loss= 'CrossEntropyLoss', device):
#     """
#     neuralnet is an instance of the model class you intend to use
#     train_dataset and test_dataset are torch.utils.dataset objects
#     optimizer =  
#     loss = 
#     """

    
def plot_comparison(path, epoch, batch_i, mask_image, predicted_image, actual_image):
    """
    path, epoch, batch_i, mask_image, predicted_image, actual_image
    """
    plt.figure(figsize = (10,4.8))
    ax = plt.subplot(1,3,1)
    ax.set_title(f'epoch_{epoch}_rgb_batch_{batch_i}')
    image = np.array(actual_image)
    image = image.transpose((1,2,0))
    plt.imshow(image)

    ax = plt.subplot(1,3,2)
    ax.set_title(f'epoch_{epoch}_mask_batch_{batch_i}')
    image = np.array(mask_image)
    # print(image.shape)
    # image = image.transpose((1,2,0))
    plt.imshow(image)

    ax = plt.subplot(1,3,3)
    ax.set_title(f'epoch_{epoch}_predicted_mask_batch_{batch_i}')
    image = np.array(predicted_image)
    # image = image.transpose((1,2,0))
    plt.imshow(image)
    name = path + f'/epoch_{epoch}_batch_{batch_i}'
    plt.savefig(name)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #wont't work on my gpu
    ENABLE_GPU = False
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
    if not os.path.exists('plots/dataset_sample.png'):
        visualizeData(dataset= trainDataset, show=False)


    num_epochs = 5
    batch_size = 1

    # for epoch in num_epochs:

    #Creating Meta Data
    IoU_training_data = np.zeros(shape= (num_epochs,1))
    IoU_testing_data = np.zeros(shape= (num_epochs,1))
    rmse_train = np.zeros(shape= (num_epochs,1))
    rmse_test = np.zeros(shape= (num_epochs,1))
    cross_entropy_loss_train = np.zeros(shape= (num_epochs,1))
    cross_entropy_loss_test = np.zeros(shape= (num_epochs,1))
    if not os.path.exists('./plots/TrainMasks'):
            os.makedirs('./plots/TrainMasks')

    if not os.path.exists('./plots/TestMasks'):
        os.makedirs('./plots/TestMasks')
    
    trainData = torch.utils.data.DataLoader(dataset = trainDataset, batch_size= batch_size, shuffle= False)
    validationData = torch.utils.data.DataLoader(dataset = testDataset, batch_size = batch_size, shuffle = False)
    network = UNet(n_classes=3)
    if ENABLE_GPU:
        network.to(device) 
    optimizer = optim.Adam(network.parameters(),lr= 1e-4)
    
    #Starting epochs
    for epoch in range(num_epochs):
        print(f'Start of Epoch {epoch+1}/{num_epochs}')
        """although we have 2 classes but those classes are labelled as 1 and 2 in the mask images. 
        So if n_classes =2 , loss function would by default consider 0 and 2, hence would give an out of bounds error"""
        #training loop
        for batch_i, batch in enumerate(trainData):
            rgb_batch = batch['rgb_image']/255
            depth_batch = batch['depth_image']/255
            mask_batch = torch.squeeze(input= batch['mask_image'].type(torch.LongTensor), dim= 1)
            if ENABLE_GPU:
                rgb_batch = rgb_batch.to(device)
                depth_batch = depth_batch.to(device)
                mask_batch = mask_batch.to(device)
            logits = network.forward(rgb_batch, depth_batch)
            cross_entropy_loss = nn.CrossEntropyLoss()
            cross_entropy_loss_value = cross_entropy_loss(logits, mask_batch)
            prediction = torch.argmax(logits, dim= 1)
                        
            optimizer.zero_grad()
            grad = cross_entropy_loss_value.backward()   
            optimizer.step()

            r = np.random.randint(0, batch_size)
            plot_comparison(path = 'plots/TrainMasks',epoch= epoch, batch_i = batch_i,
                            mask_image= mask_batch[r] , 
                            predicted_image= prediction[r],
                            actual_image = rgb_batch[r])
            if batch_i == 0:
                break

        cross_entropy_loss_train[epoch,] = cross_entropy_loss_value.detach()

        #testing loop
        for batch_i, batch in enumerate(validationData):
            with torch.no_grad():
                rgb_batch = batch['rgb_image']/255
                depth_batch = batch['depth_image']/255
                mask_batch = torch.squeeze(input= batch['mask_image'].type(torch.LongTensor), dim=1)
                if ENABLE_GPU:
                    rgb_batch = rgb_batch.to(device)
                    depth_batch = depth_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    
                logits = network.forward(rgb_batch, depth_batch)
                cross_entropy_loss = nn.CrossEntropyLoss()
                cross_entropy_loss_value = cross_entropy_loss(logits, mask_batch)
                prediction = torch.argmax(logits, dim= 1)

            r = np.random.randint(0,batch_size)
            plot_comparison(path = 'plots/TestMasks',epoch= epoch, batch_i = batch_i,
                            mask_image= mask_batch[r] , 
                            predicted_image= prediction[r],
                            actual_image = rgb_batch[r])

            if batch_i == 0:
                break

        cross_entropy_loss_test[epoch,] = cross_entropy_loss_value

        IoU_training_data[epoch,] = 0
        IoU_testing_data[epoch,] = 0
        rmse_train[epoch,] = 0
        rmse_test[epoch,] = 0
            
        print(f'Training Loss: {cross_entropy_loss_train[epoch]} and Testing Loss: {cross_entropy_loss_test[epoch]}')  
        print(f'End of Epoch {epoch+1}/{num_epochs}')
            



if __name__ == '__main__': main()

