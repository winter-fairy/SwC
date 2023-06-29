# This is a sample Python script.
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.utils.data_util import MyDataset


def process_data():
    data_path = 'Data/raw/train_10000.csv'
    save_path = 'Data/processed/train_10000_mean.csv'
    # 读取CSV数据集
    data = pd.read_csv(data_path)

    # 分离特征和标签
    features = data.drop(['sample_id', 'label'], axis=1)
    labels = data['label']

    # 标准化特征数据
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 填充缺失值
    filled_features = pd.DataFrame(scaled_features, columns=features.columns)
    filled_features.fillna(filled_features.mean(), inplace=True)

    # 合并特征和标签
    processed_data = pd.concat([filled_features, labels], axis=1)

    # 打印处理后的数据
    print(len(processed_data))
    processed_data.to_csv(save_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_data()


