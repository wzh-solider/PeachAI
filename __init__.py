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

# 判断splitted_training_data文件夹是否存在，如果不存在就新建一个
if not os.path.exists('data/splitted_training_data'):
    os.makedirs('data/splitted_training_data')


# 定义一个函数，来拆分训练集、验证集
def create_train_eval():
    '''
    划分训练集和验证集
    '''
    train_dir = train_parameters['train_image_dir']
    eval_dir = train_parameters['eval_image_dir']
    train_list_path = train_parameters['train_list_dir']
    train_data_dir = train_parameters['train_data_dir']

    print('creating training and eval images')
    # 如果文件夹不存在，建立相应的文件夹
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

        # 打开txt文件，分割数据
    file_name = train_list_path
    f = open(file_name, 'r')
    # 按行读取数据
    lines = f.readlines()
    f.close()

    for i in range(len(lines)):
        # 将每行数据按照空格分割成2部分，并取第一部分的路径名和图像文件名，例如:R0/1.png
        img_path = lines[i].split('\t')[0]
        # 取第二部分的标签，例如:R0
        class_label = lines[i].split('\t')[1].strip('\n')
        # 每8张图片取一个做验证数据,其他用于训练
        if i % 8 == 0:
            # 把目录和文件名合成一个路径
            eval_target_dir = os.path.join(eval_dir, class_label)
            # 将总的文件路径与当前图像的文件名合到一起，实际就是得到训练集图像所在的文件夹下的图像名
            eval_img_path = os.path.join(train_data_dir, img_path)
            if not os.path.exists(eval_target_dir):
                os.mkdir(eval_target_dir)
                # 将图片复制到验证集指定标签的文件夹下
            shutil.copy(eval_img_path, eval_target_dir)
        else:
            train_target_dir = os.path.join(train_dir, class_label)
            train_img_path = os.path.join(train_data_dir, img_path)
            if not os.path.exists(train_target_dir):
                os.mkdir(train_target_dir)
            shutil.copy(train_img_path, train_target_dir)
    print('划分训练集和验证集完成！')


# 制作数据集，如果已经做好了，就请将代码注释掉
# create_train_eval()

class PeachDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练、验证和测试数据集
        """
        super(PeachDataset, self).__init__()
        train_image_dir = train_parameters['train_image_dir']  # 训练集的路径
        eval_image_dir = train_parameters['eval_image_dir']
        test_image_dir = train_parameters['test_image_dir']

        '''         '''
        # transform数据增强函数，这里仅对图片的打开方式进行了转换
        # 这里用Transpose()将图片的打开方式(宽, 高, 通道数)更改为PaddlePaddle读取的方式是(通道数, 宽, 高)
        mean = [127.5, 127.5, 127.5]  # 归一化，均值
        std = [127.5, 127.5, 127.5]  # 归一化，标注差
        transform_train = T.Compose([T.ColorJitter(0.4, 0.4, 0.4, 0.4)
                                        , T.Resize(size=(224, 224))
                                        , T.Transpose()
                                        , T.Normalize(mean, std)
                                     ])
        transform_eval = T.Compose([T.Resize(size=(224, 224))
                                       , T.Transpose()
                                       , T.Normalize(mean, std)
                                    ])
        transform_test = T.Compose([T.Resize(size=(224, 224))
                                       , T.Transpose()
                                       , T.Normalize(mean, std)
                                    ])

        '''         
        # 参考API：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html#about-transforms
        #这里用Transpose()将图片的打开方式(宽, 高, 通道数)更改为PaddlePaddle读取的方式是(通道数, 宽, 高)
        # ColorJitter 随机调整图像的亮度，对比度，饱和度和色调。
        # hflip 对输入图像进行水平翻转。        
        # Normalize 归一化。mean = [127.5, 127.5, 127.5]，std = [127.5, 127.5, 127.5]
        # RandomHorizontalFlip 基于概率来执行图片的水平翻转。
        # RandomVerticalFlip 基于概率来执行图片的垂直翻转。
        mean = [127.5, 127.5, 127.5] # 归一化，均值
        std = [127.5, 127.5, 127.5] # 归一化，标注差 
        transform_train = T.Compose([T.Resize(size=(224,224)), 
                                     T.Transpose(),                                
                                     T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                     T.RandomHorizontalFlip(prob=0.5,),
                                     T.RandomVerticalFlip(prob=0.5,),
                                     T.Normalize(mean, std)])
        transform_eval = T.Compose([T.Resize(size=(224,224)), T.Transpose()])
        transform_test = T.Compose([T.Resize(size=(224,224)), T.Transpose()])
        '''

        # 飞桨推荐使用 paddle.io.DataLoader 完成数据的加载，生成一个可以加载数据的迭代器
        # 参考API:https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#cn-api-fluid-io-dataloader
        # 加载训练集，train_data_folder 是一个迭代器
        # 参考API：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/DatasetFolder_cn.html#datasetfolder
        train_data_folder = DatasetFolder(train_image_dir, transform=transform_train)
        # 加载验证集，eval_data_folder 是一个迭代器
        eval_data_folder = DatasetFolder(eval_image_dir, transform=transform_eval)
        # 加载测试集，test_data_folder 是一个迭代器
        test_data_folder = DatasetFolder(test_image_dir, transform=transform_test)
        self.mode = mode
        if self.mode == 'train':
            self.data = train_data_folder
        elif self.mode == 'eval':
            self.data = eval_data_folder
        elif self.mode == 'test':
            self.data = test_data_folder

    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = np.array(self.data[index][0]).astype('float32')

        label = np.array([self.data[index][1]]).astype('int64')

        return data, label

    # 返回整个数据集的总数
    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)


# 用自定义的PeachDataset类，加载自己的数据集
train_dataset = PeachDataset(mode='train')
val_dataset = PeachDataset(mode='eval')
test_dataset = PeachDataset(mode='test')

# DataLoader 示例代码

# 加载库
import cv2 as cv  # 使用 OpenCV

print("opencv 版本号为：" + cv.__version__)  # 查看版本号
# 事实上在使用 OpenCV之前应该安装该类库，但是由于使用了 AI-Studio，所以系统已经替开发者预先安装好了： opencv-python 4.1.1.26
from matplotlib import pyplot as plt  # 在该页面画图

# %matplotlib inline

# 构造一个 DataLoader
test_loader = DataLoader(test_dataset,
                         batch_size=2,
                         shuffle=True,
                         drop_last=True,
                         num_workers=2)

# 使用 DataLoader 来遍历数据集
for mini_batch in test_loader:  # 从 DataLoader 中获取 mini_batch
    print("mini_batch 的类型为：" + str(type(mini_batch)))
    pic_list = mini_batch[0]  # 图片数据
    label_list = mini_batch[1]  # 标记
    print("mini_batch 的大小为：" + str(len(pic_list)))

    # 将图片显示转化为 numpy 格式，并且将内部的数字设置为 整数类型
    pic_1 = pic_list[0]
    pic_2 = pic_list[1]
    arr1 = np.asarray(pic_1, dtype=np.float64)
    print(arr1.shape)
    arr2 = np.asarray(pic_2, dtype=np.float64)
    print(arr2.shape)

    break  # 由于是示例，所以仅拿出第一个 mini_batch

# 把获取到的图片数据展示出来
# arr1 = arr1 / 255 # 把每一个像素都变到 0-1 之间
# r = arr1[0]
# g = arr1[1]
# b = arr1[2]
# img = cv.merge([r, g, b])
#
# plt.imshow(img)

# 使用内置的模型,这边可以选择多种不同网络，这里选了resnet50网络
# pretrained (bool，可选) - 是否加载在imagenet数据集上的预训练权重
model = paddle.vision.models.resnet18(pretrained=True, num_classes=4)

# 尝试不同的网络结构：MobileNetV2
# MobileNetV2参考文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/MobileNetV2_cn.html
# model = paddle.vision.models.mobilenet_v2(pretrained=True, num_classes=4)

# 使用paddle.Model完成模型的封装，将网络结构组合成一个可快速使用高层API进行训练和预测的类。
model = paddle.Model(model)

# 使用 summary 观察网络信息
model.summary(input_size=(1, 3, 224, 224), dtype='float32')

# 调用Paddle的VisualDL模块，保存信息到目录中。
# log_dir (str) - 输出日志保存的路径。
callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')

# 通过Model.prepare接口来对训练进行提前的配置准备工作，包括设置模型优化器，Loss计算方法，精度计算方法等
# 优化器API文档： https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#paddle-optimizer

# 学习率衰减策略
# 学习率衰减策略 API 文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#about-lr
scheduler_StepDecay = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=50, gamma=0.9, verbose=False)
scheduler_PiecewiseDecay = paddle.optimizer.lr.PiecewiseDecay(boundaries=[100, 1000, 4000, 5000, 6000],
                                                              values=[0.1, 0.5, 0.01, 0.005, 0.001, 0.0005],
                                                              verbose=False)

# 尝试使用 SGD、Momentum 方法
sgd = paddle.optimizer.SGD(
    learning_rate=scheduler_StepDecay,
    parameters=model.parameters())

adam = paddle.optimizer.Adam(
    learning_rate=0.01,  # 调参
    parameters=model.parameters())

model.prepare(optimizer=adam,  # adam
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

# 查看当前计算设备
device = paddle.device.get_device()
print(device)
# 使用CPU训练
device = paddle.set_device('cpu')  # or 'cpu'
print(device)

# fit API文档： https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#fit-train-data-none-eval-data-none- -size-1-epochs-1-eval-freq-1-log-freq-10-save-dir-none-save-freq-1-verbose-2-drop-last-false-shuffle-true-num-workers-0-callbacks-none

# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
# epochs：总共训练的轮数
# batch_size：一个批次的样本数量
# 如果提示内存不足，可以尝试将batch_size调低
# verbose：日志显示，0为不在标准输出流输出日志信息,1为输出进度条记录，2为每个epoch输出一行记录;1为输出进度条记录，2为每个epoch输出一行记录

model.fit(train_dataset,
          val_dataset,
          epochs=1,
          batch_size=2,
          callbacks=callback,
          verbose=1)

# 模型评估
# 对于训练好的模型进行评估操作可以使用 model.evaluate 接口；操作结束后会根据 prepare 接口配置的 loss 和 metric 来进行相关指标计算返回。
# 评价指标参考文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html#evaluate-eval-data-batch-size-1-log-freq-10-verbose-2-num-workers-0-callbacks-none
model.evaluate(test_dataset, verbose=1)

# 模型保存
model.save('./saved_model/saved_model')  # save for training

# 预测模型
results = model.predict(test_dataset)

# 观察 result
print(type(results))  # list
print(len(results))  # len == 1

# 一行一行打印结果
for i in results[0]:
    print(i)

# 将结果用 softmax 处理后变成概率值
x = paddle.to_tensor(results[0])
m = paddle.nn.Softmax()
out = m(x)
print(out)

# 用一个字典，指名标签对应的数值
label_dic = {}
for i, label in enumerate(labels):
    label_dic[i] = label

# 预测标签结果写入predict_labels
predict_labels = []
# 依次取results[0]中的每个图片的预测数组
for result in results[0]:
    # np.argmax:返回一个numpy数组中的最大值的索引
    # 注意：索引是标签，不是返回数据的最大值
    lab_index = np.argmax(result)
    lab = label_dic[lab_index]
    predict_labels.append(lab)

# 看一下预测结果
print(predict_labels)

final_result = []
file_name_test = train_parameters['test_list_dir']
f = open(file_name_test, 'r')
# 按行读取数据
data = f.readlines()
for i in range(len(data)):
    # 将每行数据按照空格分割成2部分，并取第一部分的路径名和图像文件名，例如:R0/1.png
    img_path = data[i].split('\t')[0]
    final_result.append(img_path + ',' + str(predict_labels[i]) + '\n')

f.close()

with open('result.csv', "w") as f:
    f.writelines(final_result)
