import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        features = data.drop(['sample_id', 'label'], axis=1).values
        labels = data['label'].values
        self.features = torch.tensor(features, dtype=torch.float32)  # 特征值为浮点数
        self.labels = torch.tensor(labels, dtype=torch.long)  # 最终类别为整数

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    @staticmethod
    def process_data(data_path):
        """
        读取对应文件的数据，并进行标准化，使用均值填充
        :param data_path: 数据路径
        :return: 处理过后的数据
        """
        df = pd.read_csv(data_path)  # 读取数据集
        feature_columns = df.columns[1:-1]  # 选择需要标准化的特征列
        scaler = StandardScaler()  # 创建标准化对象
        df[feature_columns] = scaler.fit_transform(df[feature_columns])  # 拟合数据，即标准化数据
        df = df.fillna(df.mean())  # 使用均值填充 NaN 值
        return df
