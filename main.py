# This is a sample Python script.
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.utils.data_util import MyDataset


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def process_data1():
    data_path = 'Data/raw/train_10000.csv'
    processed_data = MyDataset.process_data(data_path)
    save_path = 'Data/processed/train_10000_mean.csv'
    processed_data.to_csv(save_path, index=False)


def process_data2():
    data_path = 'Data/raw/train_10000.csv'
    save_path = 'Data/processed/train_10000_KNN.csv'
    df = pd.read_csv(data_path)  # 读取数据集
    feature_columns = df.columns[1:-1]  # 选择需要标准化的特征列
    scaler = StandardScaler()  # 创建标准化对象
    df[feature_columns] = scaler.fit_transform(df[feature_columns])  # 拟合数据，即标准化数据
    imputer = KNNImputer(n_neighbors=3)  # 创建KNN填充器对象
    processed_data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)  # 使用KNN填充器填充DataFrame中的空值
    processed_data.to_csv(save_path, index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_data2()


