import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt


def same_seeds(seed):  # 固定随机种子 神经网络中参数默认是进行随机初始化的，设置随机数种子可以得到较好结果
    torch.manual_seed(seed)  # 为cpu设置种子用于生成随机数，以使得结果是确定的
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前gpu设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module. 用于生成指定随机数
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False  # cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
    torch.backends.cudnn.deterministic = True


def mean_confidence_interval(data, confidence=0.95):
    # 列表的置信区间 95%置信区间：当给出某个估计值的95%置信区间为【a，b】，
    # 可以理解为有95%的信心可以说样本的平均值介于a到b之间，而错误发生的概率为5%
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)  # scipy.stats.sem()用于计算输入数据平均值的标准误差
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h  # return均值和置信区间

#‘’‘ 额外添加MAE,RMSE,R2"
import numpy as np
from sklearn.metrics import r2_score


def calculate_metrics(confusion_matrix, true_labels, pred_labels):
    """
    计算所有评估指标：AA, Kappa, OA, MAE, RMSE, R²
    Args:
        confusion_matrix: 混淆矩阵
        true_labels: 真实标签（一维数组）
        pred_labels: 预测标签（一维数组）
    Returns:
        包含所有指标的字典
    """
    # 计算AA和各类别精度
    each_acc, average_acc = AA_andEachClassAccuracy(confusion_matrix)

    # 计算OA
    overall_acc = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # 计算Kappa
    total = np.sum(confusion_matrix)
    sum_rows = np.sum(confusion_matrix, axis=1)
    sum_cols = np.sum(confusion_matrix, axis=0)
    expected = np.sum(sum_rows * sum_cols) / total
    kappa = (overall_acc - expected) / (1 - expected)

    # 计算MAE
    mae = np.mean(np.abs(true_labels - pred_labels))

    # 计算RMSE
    rmse = np.sqrt(np.mean((true_labels - pred_labels) ** 2))

    # 计算R²
    r2 = r2_score(true_labels, pred_labels)

    return {
        'AA': average_acc,
        'Each_Acc': each_acc,
        'OA': overall_acc,
        'Kappa': kappa,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


# 保留原有AA计算函数（已存在于utils.py中）
def AA_andEachClassAccuracy2(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(np.divide(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


from operator import truediv


def AA_andEachClassAccuracy(confusion_matrix):  # 混淆矩阵
    counter = confusion_matrix.shape[0]  # 矩阵第一维的长度
    list_diag = np.diag(confusion_matrix)  # 混淆矩阵对角线元素 也就是正确分类的样本数的分布
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # 按行相加，不保持二维性 axis=1 就是预测样本数分布 np.sum就是将预测的每类样本数进行求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # 求每类的分类精度
    average_acc = np.mean(each_acc)  # 平均精度
    return each_acc, average_acc


import torch.utils.data as data


class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)  # 返回符合条件的索引
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()  # 返回一维数组
        self.x2 = self.x2.flatten()
        #        if medicinal==4 and d==2:
        #            self.train_data=self.imdb['data'][self.x1,:]
        #            self.train_labels=self.imdb['Labels'][self.x1]
        #            self.test_data=self.imdb['data'][self.x2,:]
        #            self.test_labels=self.imdb['Labels'][self.x2]

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ##(17, 17, 200, 10249)  #转置
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def sanity_check(all_set):  # 合理性检验
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 0:  # 原先是200个
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    # print('the number of class:', nclass)
    # print('the number of sample:', nsamples)
    return all_good


def flip(data):  # 扩增？
    y_4 = np.zeros_like(data)  # 构造一个矩阵y_4，其维度与矩阵data一致，并为其初始化为0
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)  # 矩阵拼接  ##对数据进行列操作 横着拼
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)  ##对数据进行行操作 竖着拼
    return Data  ##(435,435,200)


def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]  # 字符串切片
    label_key = label_file.split('/')[-1].split('.')[0]
    #data_key = 'data_HS_LR'
    #label_key = 'Label_image'
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204) #？？？？？？？？

    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):  # 统计推断包scipy中包含很多概率分布的随机变量
    alpha = np.random.uniform(*alpha_range)  # 从一个均匀分布的区域中随机采样 返回ndarray类型
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    # 正态分布 loc正态分布的均值，对应着这个分布的中心 loc=0说明这是一个以Y轴为对称轴的正态分布
    # scale正态分布的标准差，对应分布的宽度 size输出的值赋在shape里
    return alpha * data + beta * noise


def flip_augmentation(data):  # arrays tuple 0:(7, 7, 103) 1=(7, 7) 预处理
    horizontal = np.random.random() > 0.5  # True
    vertical = np.random.random() > 0.5  # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)  # 矩阵翻转
    return data


class Task(object):

    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))
        # print(len(class_folders))
        # print(self.num_classes)
        class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            # print(self.support_labels)
            # print(self.query_labels)


# Task_S指的是源域训练有LiDAR和HSI
class Task_S(object):

    def __init__(self, data, data2, num_classes, shot_num, query_num):
        self.data = data
        self.data2 = data2
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))
        class_folders_L = sorted(list(data2))
        class_list = random.sample(class_folders, self.num_classes)
        class_list_L = random.sample(class_folders_L, self.num_classes)
        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()  # 创建空字典

        self.support_datas = []
        self.support_datas_L = []
        self.query_datas_L = []
        self.query_datas = []
        self.support_labels = []
        self.support_labels_L = []
        self.query_labels = []
        self.query_labels_L = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.support_datas_L += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]
            self.query_datas_L += samples[c][shot_num:shot_num + query_num]
            self.support_labels += [labels[c] for i in range(shot_num)]
            self.support_labels_L += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            self.query_labels_L += [labels[c] for i in range(query_num)]
            # print(self.query_labels)


# 为官方源Task,为目标域训练，只有HSI
class Task_T(object):

    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))
        class_list = random.sample(class_folders, self.num_classes)
        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()  # 创建空字典

        self.support_datas = []
        # self.support_datas_L=[]
        # self.query_datas_L=[]
        self.query_datas = []
        self.support_labels = []
        # self.support_labels_L=[]
        self.query_labels = []
        # self.query_labels_L=[]
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            # self.support_datas_L+=samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]
            # self.query_datas_L+=samples[c][shot_num:shot_num + query_num]
            self.support_labels += [labels[c] for i in range(shot_num)]
            # self.support_labels_L += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            # self.query_labels_L += [labels[c] for i in range(query_num)]

            # print(self.support_labels)
            # print(self.query_labels)


class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

        def __len__(self):
            return len(self.image_datas)

        def __getitem__(self, idx):
            raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


# 定义一个LiDAR的FewShotDataset
class FewShotDataset_L(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas_L = self.task.support_datas_L if self.split == 'train' else self.task.query_datas_L
        self.labels_L = self.task.support_labels_L if self.split == 'train' else self.task.query_labels_L

    def __len__(self):
        return len(self.image_datas_L)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label


class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label


# 定义一个有关LiDAR的HBKC_dataset
class HBKC_dataset_L(FewShotDataset_L):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset_L, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_L = self.image_datas_L[idx]
        label_L = self.labels_L[idx]
        print("在HBKC_dataset中的image_L的size:", image_L.shape)
        return image_L, label_L


# Sampler
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''

    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


# dataloader
# 为源域训练，有HSI和LiDAR
def get_HBKC_data_loader_S(task, num_per_class=1, split='train', shuffle=False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task, split=split)
    dataset_L = HBKC_dataset_L(task, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num,
                                       shuffle=shuffle)  # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)  # query set

    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    loader_L = DataLoader(dataset_L, batch_size=num_per_class * task.num_classes, sampler=sampler)
    print("在经过DataLoader之后loader和loader_L是一样的吗", loader_L == loader)
    return loader, loader_L


# 为目标域训练只有HSI,无LiDAR

def get_HBKC_data_loader(task, num_per_class=1, split='train', shuffle=False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num,
                                       shuffle=shuffle)  # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)  # query set

    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader


def get_HBKC_data_loader_T(task, num_per_class=1, split='train', shuffle=False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task, split=split)
    # dataset_L = HBKC_dataset_L(task, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num,
                                       shuffle=shuffle)  # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)  # query set

    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    # loader_L=DataLoader(dataset_L, batch_size=num_per_class*task.num_classes, sampler=sampler)
    return loader


def classification_map(map, groundTruth, dpi, savePath):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1] * 2.0 / dpi, groundTruth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi=dpi)

    return 0
