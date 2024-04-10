# os ： OS模块提供了非常丰富的方法用来处理文件和目录。
# sys：sys模块提供了一系列有关Python运行环境的变量和函数。
# shutil:用于文件拷贝的模块
# numpy：numpy 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
# random：Python中的random模块用于生成随机数。
# paddle.vision.datasets：该模块包含数据加载的相关函数，比如可以用来加载常用的数据集等，如mnist。
# paddle.vision.transforms:该模块包含对图像进行转换的函数，比如把HWC格式的图片，转变成CHW模式的输入张量。也包含飞桨框架对于图像预处理的方式，可以快速完成常见的图像预处理，如调整色调、对比度，图像大小等；
# paddle.io.Dataset:高模块包含了飞桨框架数据加载方式，可以“一键”完成数据的批加载与异步加载。
import os
import sys
import shutil
import numpy as np
import paddle
import random
from paddle.io import Dataset, DataLoader
from paddle.vision.datasets import DatasetFolder, ImageFolder
from paddle.vision import transforms as T

# '''
# 参数配置：
# 'train_data_dir'是提供的经增强后的原始训练集；
# 'test_image_dir'是提供的原始测试集；
# 'train_image_dir'和'eval_image_dir'是由原始训练集经拆分后生成的实际训练集和验证集
# 'train_list_dir'和'test_list_dir'是生成的txt文件路径
# 'saved_model' 存放训练结果的文件夹
# '''
train_parameters = {
    'train_image_dir': './data/splitted_training_data/train_images',
    'eval_image_dir': './data/splitted_training_data/eval_images',
    'test_image_dir': './data/enhancement_data/test',
    'train_data_dir': './data/enhancement_data/train',
    'train_list_dir': './data/enhancement_data/train.txt',
    'test_list_dir': './data/enhancement_data/test.txt',
    'saved_model': './saved_model/'
}

# 数据集的4个类别标签
labels = ['R0', 'B1', 'M2', 'S3']
labels.sort()

# 准备生成训练集文件名、标签名的txt文件
write_file_name = train_parameters['train_list_dir']

# 以写方式打开write_file_name文件
with open(write_file_name, "w") as write_file:
    # 针对不同的分类标签分别录入
    for label in labels:
        # 建立空列表，用于保存图片名
        file_list = []
        # 用于找到该标签路径下的所有图片.
        train_txt_dir = train_parameters['train_data_dir'] + '/' + label + '/'

        for file_name in os.listdir(train_txt_dir):
            dir_name = label
            temp_line = dir_name + '/' + file_name + '\t' + label + '\n'  # 例如："B1/101.png	B1"
            write_file.write(temp_line)

# 准备生成测试集文件名、标签名的txt文件
write_file_name = train_parameters['test_list_dir']

# 以写方式打开write_file_name文件
with open(write_file_name, "w") as write_file:
    # 针对不同的分类标签分别录入
    for label in labels:
        # 建立空列表，用于保存图片名
        file_list = []
        # 用于找到该标签路径下的所有图片.
        train_txt_dir = train_parameters['test_image_dir'] + '/' + label + '/'

        for file_name in os.listdir(train_txt_dir):
            dir_name = label
            temp_line = dir_name + '/' + file_name + '\t' + label + '\n'  # 例如："B1/101.png	B1"
            write_file.write(temp_line)
