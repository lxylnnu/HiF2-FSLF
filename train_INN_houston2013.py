
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse

import pickle

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

import time
import utils
import models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from FeatureExtract2_INN import Restormer_Encoder, FusionNet

# np.random.seed(1337)
##############################尝试将Trento数据集作为源数据集放入到DCFSL中
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")  # 创建对象
parser.add_argument("-f", "--feature_dim", type=int, default=1600)
parser.add_argument("-c", "--src_input_dim", type=int, default=144)
parser.add_argument("-d", "--tar_input_dim", type=int, default=144)  # Houston2013=133.#MUUFL=64#Trento=63
parser.add_argument("-n", "--n_dim", type=int, default=100)
parser.add_argument("-w", "--class_num", type=int, default=7)
parser.add_argument("-s", "--shot_num_per_class", type=int, default=1)
parser.add_argument("-b", "--query_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=10000)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)#0.0001
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
# target
parser.add_argument("-m", "--test_class_num", type=int, default=7)
parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5, help='5 4 3 2 1')
clip_grad_norm_value = 0.01
args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

TEST_CLASS_NUM = args.test_class_num  # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class  # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')


_init_()

# load source domain data set
print("------------------------------------------------------------------------------------导入Hutson2013的HSI")
with open(os.path.join('./datasets_Huston2013/LeftUp_175_477/LeftUp_175_477', 'Huston2013_HSI.pickle'),
          'rb') as handle:
    source_imdb = pickle.load(handle)
print("------------------------------------------------------------------------------------导入Huston2013的LiDAR")
with open(os.path.join('./datasets_Huston2013/LeftUp_175_477/LeftUp_175_477', 'Huston2013_LiDAR.pickle'),
          'rb') as handle:
    source_imdb_L = pickle.load(handle)
data_train = source_imdb['data']
labels_train = source_imdb['Labels']
data_train_L = source_imdb_L['data']
labels_train_L = source_imdb_L['Labels']
print("查看HSI的data的尺寸", data_train.shape)
print("查看HSI的label的尺寸", labels_train.shape)
print("查看LiDAR的data的尺寸", data_train_L.shape)
print("查看LiDAR的label的尺寸", labels_train_L.shape)
keys_all_train_L = sorted(list(set(labels_train_L)))
keys_all_train = sorted(list(set(labels_train)))
print("打印labels_train的元素类别",
      keys_all_train)
print("打印LiDAR的labels_train的元素类别", keys_all_train_L)

label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print("查看label_encoder_train这个字典集合:", label_encoder_train)

train_set = {}
i = 0
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
    i = i + 1
i = 0
train_set_L = {}
for class_, path in zip(labels_train_L, data_train_L):
    if label_encoder_train[class_] not in train_set_L:
        train_set_L[label_encoder_train[class_]] = []
    train_set_L[label_encoder_train[class_]].append(path)
    i = i + 1
print("这个train_set做了个汇总，具体将样本划分到类中", train_set.keys())
data = train_set
data_L = train_set_L
del train_set
del keys_all_train
del label_encoder_train
del train_set_L

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())

print("--------------------------------得到matetrain_data(包含每类200个,注释写着做源域小样本的数据")
data = utils.sanity_check(data)  # 200 labels samples per class
data_L = utils.sanity_check(data_L)
print("Num classes of the number of class larger than 200: " + str(len(data)))
for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose
        # print("HSI的image_transpose",image_transpose.shape)

for class_ in data_L:
    for i in range(len(data_L[class_])):
        image_transpose_L = np.transpose(data_L[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data_L[class_][i] = image_transpose_L

# source few-shot classification data
metatrain_data = data  ####每类200个样本去做小样本源域
metatrain_data_L = data_L
print(len(metatrain_data.keys()), metatrain_data.keys())
del data
del data_L
# source domain adaptation data
print("------------------------------------得到source_loader(注释写着是做源域预适应的数据)")
print("这是原本Huston2013HSI的数据大小", source_imdb['data'].shape)  # (77592, 9, 9, 128)
print("这是原本Huston2013——LiDAR的数据大小", source_imdb_L['data'].shape)  # (77592, 9, 9, 128)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))  # (9, 9, 128, 77592)
print("在将Hutson2013的HSI数据做了通道交换之后的尺寸", source_imdb['data'].shape)  # (9, 9, 128, 77592)
source_imdb_L['data'] = source_imdb_L['data'].transpose((1, 2, 3, 0))  # (9, 9, 128, 77592)
print("在Hutson2013的LiDAR数据做了通道交换之后的尺寸", source_imdb_L['data'].shape)  # (9, 9, 128, 77592)
# print(source_imdb['Labels'])
###在经过matcifar之后，反正就是有选择的对Chikusei数据分成训练和测试数据集。但matcifar咋分的，我还是不太懂？
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_dataset_L = utils.matcifar(source_imdb_L, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size= 128, shuffle=True, num_workers=0)       #这里是128
source_loader_L = torch.utils.data.DataLoader(source_dataset_L, batch_size= 128, shuffle=True, num_workers=0)
del source_dataset, source_imdb
del source_dataset_L, source_imdb_L

print("-------------------------------------------------------------------------------------导入目标域数据集Huston2013的右半张图像")
## target domain data set
# load target domain data set
test_data = './datasets_Huston2013/HSI.mat'
test_label = './datasets_Huston2013/gt.mat'
# 打印图像的维度信息 paviaU (610, 340, 103)
Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)
Data_Band_Scaler = Data_Band_Scaler[:175, 477:953, :]
GroundTruth = GroundTruth[:175,477:953]
# 得到经过归一化后的目标域数据集
print("打印目标域数据集的尺寸大小", Data_Band_Scaler.shape)


# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print("打印目标域数据集的尺寸大小", Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    print("在经过utils.flip之后的data_band_scaler", data_band_scaler.shape)
    groundtruth = utils.flip(GroundTruth)

    del Data_Band_Scaler
    del GroundTruth
    #HalfWidth=4
    HalfWidth =4

    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]

    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)

    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample(指的是背景类别这一数量吗?)', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)

    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)
    # 遍历每个类别
    for i in range(m):
        # 获取当前类别的所有样本的索引
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        # 对索引进行随机排序，以实现随机打乱的效果
        np.random.shuffle(indices)
        # 每个类别的带标签样本数量（即不进行数据增强的样本数量）为shot_num_per_class
        nb_val = shot_num_per_class
        # 将前nb_val个索引作为当前类别的训练样本，存入train字典中，键为类别i，值为该类别的训练样本索引列表
        train[i] = indices[:nb_val]
        # 将前nb_val个索引作为当前类别带有数据增强的训练样本，存入da_train字典中，键为类别i，值为该类别的带有数据增强的训练样本索引列表
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):  # 根据上面计算得到的数据增强样本数量进行循环
            da_train[i] += indices[:nb_val]  # 将当前类别的带标签样本复制若干次，存入带有数据增强的训练样本列表中
        # 将剩余的索引作为当前类别的测试样本，存入test字典中，键为类别i，值为该类别的测试样本索引列表
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]  # 将train字典中的值合并到train_indices列表中（即训练样本索引列表）
        test_indices += test[i]  # 将test字典中的值合并到test_indices列表中（即测试样本索引列表）
        da_train_indices += da_train[i]  # 将da_train字典中的值合并到da_train_indices列表中（即带有数据增强的训练样本索引列表）
    # 对测试样本索引列表进行随机排序，以实现随机打乱的效果
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # the number of train_indices: 45
    print('the number of test_indices:', len(test_indices))  # the number of test_indices: 42731
    print('the number of train_indices after data argumentation:',
          len(da_train_indices))  # the number of train_indices after data argumentation: 1800
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    # 定义一个3D数组，存储所有样本的数据，其中第一维为9（左右各加半宽度的3），第二维为9（上下各加半宽度的3），第三维为100（假设有100个频带），第四维为样本数量，即nTrain + nTest
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,100,n)
    # 定义一个数组，存储所有样本的标签
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    # 定义一个数组，存储所有样本的类别信息，用于分类任务
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)
    # 将训练样本和测试样本的索引合并成一个列表，用于后续的数据读取和标签提取
    RandPerm = train_indices + test_indices
    # 将合并后的索引列表转换为numpy数组，方便后续操作
    RandPerm = np.array(RandPerm)
    # 遍历所有样本，从数据中提取出样本的图像数据和对应的标签，并存储到imdb字典中
    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[
                                                                                    RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)
    # 将标签信息进行转换，从1-16变为0-15的形式，方便后续处理和计算
    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    # 根据训练集和测试集的样本数量，给每个样本设定类别信息，其中训练集为1，测试集为3*nTest（假设每个测试样本的类别为3）
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')
    # 使用定义的imdb字典创建训练数据集，并使用torch的DataLoader函数创建数据加载器，用于后续的数据读取和模型训练过程
    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False,
                                               num_workers=0)
    del train_dataset
    # 使用定义的imdb字典创建测试数据集，并使用torch的DataLoader函数创建数据加载器，用于后续的模型测试过程
    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)#之前是100
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    print("----------------------------------------数据增强加噪声")
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, \
        class_num=class_num, shot_num_per_class=shot_num_per_class)
    train_datas, train_labels = next(train_loader.__iter__())
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)  # 数据归一化

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


class Mapping_L(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping_L, self).__init__()
        #self.preconv = nn.Conv2d(2, out_dimension, 1, 1, bias=False)  ###在MUUFL的LiDAR中有2个通道
        self.preconv = nn.Conv2d(1, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)  # 数据归一化

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


DIDF_Encoder = Restormer_Encoder()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = DIDF_Encoder
        self.final_feat_dim = 160  # 64+32
        #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
        self.classifier = nn.Linear(in_features=1600, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.target_mapping_L = Mapping_L(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping_L = Mapping_L(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, cf=0, mode="HSI", domain='source'):  # x
        #
        ss = x.shape[1]
        if domain == 'target':
            if ss == 63 or ss == 144 or ss == 64:
                x = self.target_mapping(x)  # (45, 100,9,9)
            else:
                x = self.target_mapping_L(x)
        elif domain == 'source':
            if ss == 63 or ss == 144 or ss == 64:
                x = self.target_mapping(x)  # (45, 100,9,9)
            else:
                x = self.target_mapping_L(x)

        if mode == "HSI":
            base_feature, detial_feature, shallow_feature = self.feature_encoder(x,mode)
            return base_feature, detial_feature , shallow_feature
        else:
            detial_feature_L = self.feature_encoder(x,mode)
            return detial_feature_L


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


crossEntropy = nn.CrossEntropyLoss().cuda()  # 交叉熵损失函数
domain_criterion = nn.BCEWithLogitsLoss().cuda()
################################尝试打印特征经过每一层的维度变化
feature_encoder = Network()


# summary(feature_encoder,(128,9,9),device='cpu')

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


# run 10 times
nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet,CLASS_NUM])#原先这里是CLASS_NUM
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

# 初始化惩罚因子
penalty_factor = 1.0

if __name__ == "__main__":
    # 初始化模型
    model = Restormer_Encoder()
    model.eval()

    # 计算复杂度
    print("模型关键组件复杂度分析:")
    model.calculate_complexity()  # 输入形状为(100, 9, 9)，根据实际情况调整


# 训练循环
best_f_loss = float('inf')
optim_step = 20
optim_gamma = 0.5
seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(1330)#seeds[iDataSet]
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    feature_fusion = FusionNet()
    domain_classifier = models.DomainClassifier()
    random_layer = models.RandomLayer([args.feature_dim, args.class_num], 1024)

    feature_encoder.apply(weights_init)
    feature_fusion.apply(weights_init)
    domain_classifier.apply(weights_init)

    feature_encoder.cuda()  # GPU
    feature_fusion.cuda()
    domain_classifier.cuda()
    random_layer.cuda()  # Random layer

    feature_encoder.train()
    feature_fusion.train()
    domain_classifier.train()
    # optimizer  #参数优化器
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),
                                             lr=args.learning_rate)  # 特征编码优化器 能根据计算得来的梯度更新参数 lr学习率 Adam算法
    feature_fusion_optim = torch.optim.Adam(feature_fusion.parameters(), lr=args.learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)
    # 优化器调用方法
    #scheduler1 = torch.optim.lr_scheduler.StepLR(feature_fusion_optim, step_size=optim_step, gamma=optim_gamma)
    #scheduler2 = torch.optim.lr_scheduler.StepLR(feature_encoder_optim, step_size=optim_step, gamma=optim_gamma)
    # optimizer.step()
    for i in range(args.episode):
        damping = (1 - i / EPISODE)
    print("Training...")
    #对源域和目标域的小样本损失进行画图
    s_loss=[]
    t_loss=[]
    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []
    source_iter_L = iter(source_loader_L)
    source_iter = iter(source_loader)  # 生成迭代器 source_loader=source_dataset
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(10000):

        # get domain adaptation data from  source domain and target domain
        try:  # 先执行try模块，发现错误执行except
            source_data, source_label = next(source_iter)
            source_data_L, source_label_L = next(source_iter_L)
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = next(source_iter)

        try:
            target_data, target_label = next(target_iter)
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = next(target_iter)

        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''  # 源域数据集的小样本说明
            # get few-shot classification samples 获取小样本分类样本
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            # 添加LiDAR的
            task_L = utils.Task(metatrain_data_L, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)
            support_dataloader_L = utils.get_HBKC_data_loader(task_L, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                              shuffle=False)
            query_dataloader_L = utils.get_HBKC_data_loader(task_L, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                            shuffle=True)
            # sample datas 样本数据
            supports, support_labels = next(support_dataloader.__iter__())
            # print("query_labels的大小：",query_labels.shape)
            supports_L, support_labels_L = next(support_dataloader_L.__iter__())
            querys_L, query_labels_L = next(query_dataloader_L.__iter__())

            # 仿照CDDFuse
            feature_fusion.zero_grad()
            feature_fusion_optim.zero_grad()
            # calculate features 特征预测
            #将HSI和LiDAR分为两部分，HSI走RestormerBlock, LiDAR走INN
            support_features_B_H, support_features_D_H, support_shallow = feature_encoder( supports.cuda() , mode = "HSI")  # torch.Size([409, 32, 7, 3, 3])
            support_features_D_L  = feature_encoder(supports_L.cuda() , mode = "LiDAR")
            # support_features_B, support_features_D, support_features_B_L, support_features_D_L, output = feature_encoder(supports.cuda(),supports_L.cuda())

            query_features_B_H, query_features_D_H, query_shallow  = feature_encoder(querys.cuda(), mode="HSI")
            query_features_D_L = feature_encoder(querys_L.cuda(), mode="LiDAR")
            #print("打印一下经过RestormerBlock后特征分解为低高频特征大小：",support_features_B_H.shape:torch.Size([10, 100, 4, 4])
            # 特征融合部分
            support_features, support_outputs = feature_fusion(support_features_B_H, support_features_D_H,support_features_D_L, support_shallow)

            query_features, query_outputs = feature_fusion(query_features_B_H, query_features_D_H ,query_features_D_L, query_shallow)
            target_features_B, target_features_D, target_shallow = feature_encoder(target_data.cuda(),cf=1,mode="HSI",domain='target')
            target_features, target_outputs = feature_fusion(target_features_B,target_features_D,target_features_D, target_shallow, domain='target')
            # Prototype network 模型网络
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features

            # fsl_loss
            #print("在源域进行欧氏距离度量之前的query和support是有什么区别：",query_features.shape,support_proto.shape)
            logits = euclidean_metric(query_features, support_proto)  # logits: torch.Size([5, 5])
            # print("logits:",logits.shape)
            query_labels=query_labels.long()
            f_loss = crossEntropy(logits, query_labels.cuda())
            # 在每个epoch结束后，记录损失值
            s_loss.append(f_loss.item())

            ''' domain adaptation '''
            # calculate domain adaptation loss
            features = torch.cat([support_features, query_features, target_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)
            # set label: source 1; target 0
            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).cuda()
            domain_label[:supports.shape[0] + querys.shape[0]] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)

            # total_loss = fsl_loss + domain_loss
            loss = 0.001*f_loss + damping*domain_loss# 0.01

            # Update parameters
            feature_fusion.zero_grad
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            feature_fusion_optim.step()
            domain_classifier_optim.step()
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()  # 每行最大距离求和
            total_num += querys.shape[0]  # 用于求精度

        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,
                              QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)

            # sample datas
            supports, support_labels = next(iter(support_dataloader)) # (5, 100, 9, 9)
            querys, query_labels = next(iter(query_dataloader)) # (75,100,9,9)

            # calculate features
            support_features_B, support_features_D, support_shallow  = feature_encoder(supports.cuda(),cf=1, mode="HSI",domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_features_B, query_features_D, query_shallow = feature_encoder(querys.cuda(),cf=1, mode="HSI",domain='target')  # torch.Size([409, 32, 7, 3, 3])
            source_features_B, source_features_D, source_shallow = feature_encoder(source_data.cuda(),cf=1, mode="HSI")  # torch.Size([409, 32, 7, 3, 3])
            #特性融合
            support_features, support_outputs = feature_fusion(support_features_B,support_features_D,support_features_D,support_shallow, domain='target')
            query_features, query_outputs = feature_fusion(query_features_B, query_features_D, query_features_D, query_shallow, domain='target')
            source_features, source_outputs = feature_fusion(source_features_B,source_features_D,source_features_D,source_shallow ,domain='target')
            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features
            query_labels=query_labels.long()
            f_loss = crossEntropy(logits, query_labels.cuda())  # 交叉熵损失函数
            # 在每个epoch结束后，记录损失值
            t_loss.append(f_loss.item())
            '''domain adaptation'''
            features = torch.cat([support_features, query_features, source_features], dim=0)  # 竖着拼
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).cuda()
            domain_label[supports.shape[0] + querys.shape[0]:] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)  # , label_logits
            domain_loss = domain_criterion(domain_logits, domain_label)

            # total_loss = fsl_loss + domain_loss
            loss = 0.001*f_loss + damping * domain_loss  # 0.01 0.5=78;0.25=80;0.01=80

            # Update parameters
            feature_fusion.zero_grad
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            ###nn.utils.clip_grad_norm_(
                #feature_encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            #nn.utils.clip_grad_norm_(
                #feature_fusion.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            feature_encoder_optim.step()
            feature_fusion_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
            # 更新最佳验证损失和惩罚因子
            #if f_loss < best_f_loss:
            #    best_f_loss = f_loss
            #penalty_factor = update_penalty_factor(penalty_factor,f_loss, episode)
        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print("打印出total_num：",total_num)
            print('episode {:>3d}:  domain loss: {:6.4f}, fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(
                episode + 1, \
                domain_loss.item(),
                f_loss.item(),
                total_hit / total_num,
                loss.item()))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            feature_fusion.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels =next(iter( train_loader))
            train_features_B,train_features_D, train_shallow = feature_encoder(Variable(train_datas).cuda(),cf=1, mode="HSI" , domain='target')  # (45, 160)
            train_features, _ = feature_fusion(train_features_B,train_features_D,train_features_D,train_shallow, domain='target')
            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)
            # print("打印Testing的train_features和train_labels:",train_features.shape, train_labels.shape)
            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(),
                               train_labels)  # .cpu().detach().numpy()  返回训练过程的数据记录
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]
                test_features_B, test_features_D,test_shallow  = feature_encoder(Variable(test_datas).cuda(),cf=1,mode="HSI", domain='target')  # (100, 160)
                test_features, _ = feature_fusion(test_features_B,test_features_D,test_features_D,test_shallow, domain='target')
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)
            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            # 计算评估指标
            mae = mean_absolute_error(labels, predict)
            rmse = np.sqrt(mean_squared_error(labels, predict))
            r2 = r2_score(labels, predict)

            print("counter:", counter)
            print(f'\t\tAccuracy: {total_rewards}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)')
            print(f'\t\tMAE: {mae:.4f}')
            print(f'\t\tRMSE: {rmse:.4f}')
            print(f'\t\tR²: {r2:.4f}\n')

            test_end = time.time()
            print("counter:",counter)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            feature_fusion.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(), str(
                    "checkpoints/HiF2-FSL_feature_encoder_" + "Huston2013_" + str(iDataSet) + "iter_" + str(
                        TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                print("save networks for episode:", episode + 1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                print(max(labels),max(predict))
                print(labels.shape)
                C = metrics.confusion_matrix(labels, predict)
                print(C.shape)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)  ##原先这个位置是np.float

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)

####################################################对于小样本损失画图#####################
import matplotlib.pyplot as plt

# 绘制第一个损失的曲线
plt.plot(s_loss, label='s_loss')

# 绘制第二个损失的曲线
plt.plot(t_loss, label='t_loss')

# 设置x轴和y轴的标签
plt.xlabel('Episode')
plt.ylabel('Loss')

# 设置图表的标题
plt.title('Huston2013 Loss Curves')

# 显示图例
plt.legend()
plt.savefig("./Huston2013_Loss.jpg")
# 显示绘制的图形
plt.show()

print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))


#################classification map################################

print(predict.shape[0])
for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[
                                                                                                        i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]

utils.classification_map(hsi_pic, best_G, 24,
                         "./classificationMap/Huston2013_INN_{}shot.png".format(
                             TEST_LSAMPLE_NUM_PER_CLASS))
