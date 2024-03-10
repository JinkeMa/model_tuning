# 导入torch常用模块
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as Optimum

import torchvision.datasets as Datasets
import torchvision.transforms as Transforms
import torchvision.io as tio

from torchvision.transforms import ToTensor, Lambda, Compose

import os
import cv2

# 全局变量
data_path = {
    'train' : 'F:/model_tuning/data/viod/train',
    'test'  : 'F:/model_tuning/data/viod/test'
}

# 读取目录下的jpg原图像与同名png掩膜图像
def read_image(data_path):
    # 判断目录是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError('文件夹不存在')
    # 读取目录下的所有jpg图像与png图像
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
    # 用torchvision库读取image_paths里所有的图像
    images = [tio.read_image(f) for f in image_paths]
    # 将Tensor作为图像展示
    print(images[0].shape)
    cv2.imshow('image', images[0].permute(1,2,0).numpy())
    cv2.waitKey(0)
    

# 自定义数据集类DataSet
class MyDataSet(Datasets.VisionDataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        super(MyDataSet, self).__init__(root_dir, transform, target_transform)

    
if __name__ == '__main__':
    read_image(data_path['train'])