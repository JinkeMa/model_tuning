# 导入torch常用模块
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as Optimum

import torchvision.datasets as Datasets
import torchvision.transforms as Transforms

from torchvision.transforms import ToTensor, Lambda, Compose



import os
# 全局变量
data_path = {
    'train' : '../data/viod/train/',
    'test'  : '../data/viod/test/'
}
# 读取目录下的.json文件与对应的.jpg图像
def read_json_image(data_path):
    # 判断目录是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError('文件夹不存在')
    # 读取目录下的jpg与json文件
    img_files = os.listdir(data_path)
    


# 加载unet网络的标注数据
train_data = Datasets.ImageFolder(root='../data/viod/train/', transform=ToTensor())
test_data = Datasets.ImageFolder(root='../data/viod/test/', transform=ToTensor())
# 加载训练集和测试集
train_loader = Data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=16, shuffle=False)