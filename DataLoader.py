import torch 
import os
import torch.nn as nn
import torch.optim as optim
import math
import os
import pandas as pd 
import numpy as np
from skimage import io, transform
from torchvision import transforms, utils

torch.manual_seed(0)

class COSD_Dataset_Loader(torch.utils.data.Dataset):
    def __init__(self, rgb_dir_path, depth_dir_path, csv_file_path, mask_dir_path = None, data_type = 'i', transform = None):
        """
            mask_dir_path is required when data_type is 'o'
            data_type can take on values 'i' and 'o'
        """
        self.rgb_dir = rgb_dir_path
        self.depth_dir = depth_dir_path
        self.transform = transform
        self.df = pd.read_csv(csv_file_path, index_col = 0)
        self.mask_dir = mask_dir_path
        self.data_type = data_type



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        # print(self.df.head())
        # print(self.df.iloc[idx,0])
        rgb_img_path = os.path.join(self.rgb_dir,self.df.iloc[idx,0])
        depth_img_path = os.path.join(self.depth_dir,self.df.iloc[idx,1])
        mask_img_path = os.path.join(self.mask_dir, self.df.iloc[idx,2])
        
        mask_image = io.imread(fname= mask_img_path)
        rgb_image = io.imread(fname= rgb_img_path)
        depth_image = io.imread(fname= depth_img_path)
        classified = self.df.iloc[idx,3:]
        classified = np.array([classified])
        
        
        sample = {'rgb_image':rgb_image,
                    'depth_image':depth_image,
                    'mask_image':mask_image,
                    'classes':classified}

        if self.transform:
            sample = self.transform(sample)

        return sample


    def __len__(self):
        return len(self.df)




class ToTensor(object):
    def __call__(self, sample):
        keys = list(sample.keys())
        # print(keys)
        # print(sample[keys[0]].shape)
        classes = keys[-1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for i in keys[:-1] :
            # print(i)
            shape = sample[i].shape
            # print(len(shape), f'shape = {shape}')
            if len(shape) == 3:
                image = sample[i].transpose((2,0,1))
            else:
                # image = sample[i].reshape( (1,480,640))
                image = sample[i][np.newaxis,:,:]
                image = np.array(image, dtype = np.int32)

            sample[i] = torch.from_numpy(image)

        # print(f'classes: {sample[classes]}')
        sample[classes] = torch.from_numpy(np.array(sample[classes], dtype = np.bool))
        return sample 
        



def Generate_csv_logs_for_images(column_names, rgb_path, depth_path, mask_path, mirror_path, transparent_path):
    """
        column_names = [rgb column, depth_column, mask_column, mirror, transparent]
    """
    rgb_images = pd.Series(os.listdir(rgb_path), name = column_names[0])
    depth_images = pd.Series(os.listdir(depth_path), name = column_names[1])
    mask_images = pd.Series(os.listdir(depth_path), name = column_names[2])

    df = pd.DataFrame([rgb_images, depth_images, mask_images]).T
    # print(df.head())
    
    with open(mirror_path) as mirrors:
        lines = mirrors.readlines()
        mirror_labels = list()
        for line in lines:
            remove_newline = line.replace('\n','')
            mirror_labels.append(remove_newline)


    with open(transparent_path) as trans:
        lines = trans.readlines()
        transparent_labels = list()
        for line in lines:
            remove_newline = line.replace('\n','')
            transparent_labels.append(remove_newline)

    # print(df['depth_names'][mirror_labels])

    df[column_names[3]] = df['depth_names'].map(lambda x : True if x in mirror_labels else False)  
    df[column_names[4]] = df['depth_names'].map(lambda x : True if x in transparent_labels else False)

    df.to_csv('train/csv_file.csv')

    # print(len(mirror_labels))
    # print(len(transparent_labels))
    # print(df.head(5))
    # print(df[column_names[2]].value_counts())
    # print(df[column_names[3]].value_counts())







if os.path.isfile('train/csv_file.csv') == False:
    Generate_csv_logs_for_images(column_names= ['rgb_names', 'depth_names', 'mask_names', 'Mirror','Transparent']
                                ,rgb_path = 'train/rgb'
                                ,depth_path = 'train/depth'
                                ,mask_path = 'train/mask'
                                ,mirror_path = 'train/train_mirror.txt'
                                ,transparent_path= 'train/train_trans.txt'
                                )

dataset_input = COSD_Dataset_Loader(rgb_dir_path= 'train/rgb'
                            , depth_dir_path= 'train/depth'
                            , csv_file_path= 'train/csv_file.csv'
                            , mask_dir_path= 'train/mask'
                            , data_type= 'i'
                            , transform = transforms.Compose([ToTensor()]))

dataset_output = COSD_Dataset_Loader(rgb_dir_path= 'train/rgb'
                                    ,depth_dir_path= 'train/depth'
                                    ,csv_file_path= 'train/csv_file.csv'
                                    ,mask_dir_path= 'train/mask'
                                    ,data_type= 'o'
                                    ,transform= transforms.Compose([ToTensor()])
)

# print(dataset)
# # print(len(dataset))
# for i in range(3):
#     sample = dataset[i]
#     # print(i, sample['rgb_image'].shape,sample['depth_image'].shape, sample['classes'].shape)
#     print(i, sample['mask_image'].shape, sample['classes'].shape)


# dataloader_output = torch.utils.data.DataLoader(dataset_output, batch_size= 5, shuffle = True)
# # # print(dataloader)
# dataloader_input = torch.utils.data.DataLoader(dataset_input, batch_size= 5, shuffle = True)

# # for batch_i , sample_batch in enumerate(dataloader_output):
# #     print(batch_i, sample_batch['mask_image'].size(), sample_batch['classes'].size)
# #     break

# print(len(dataset_input))

# for batch_i, sample_batch in enumerate(dataloader_input):
#     print(batch_i, sample_batch['rgb_image'].size(), sample_batch['depth_image'].size(), sample_batch['mask_image'].size(),sample_batch['classes'].size())
#     network_input = torch.cat((sample_batch['rgb_image'], sample_batch['depth_image']), dim = 1)
#     print(network_input.size())
#     #the python file is successfully working
#     break