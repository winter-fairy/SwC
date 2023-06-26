import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# 读取CSV文件
data = pd.read_csv('../../Data/train_10000.csv')

feature_columns = data.columns[1:-1]  # 选择需要标准化的特征列
scaler = StandardScaler()  # 创建标准化对象
data[feature_columns] = scaler.fit_transform(data[feature_columns])  # 拟合数据，即标准化数据
data = data.fillna(data.mean())  # 使用均值填充 NaN 值

# 提取特征列
features = data.iloc[:, 1:107]

# 计算相关性矩阵
correlation_matrix = features.corr()

# # 打印相关性矩阵
# print(correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='viridis', annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
plt.title('Correlation Matrix')
plt.show()

# 获取相关性最大的10对特征
n = 10  # 取前10个最大相关性
correlation_values = correlation_matrix.unstack().sort_values(ascending=True)
top_correlations = correlation_values[correlation_values < 1].head(n)

# 打印相关性最大的10对特征
for (feature1, feature2), correlation in top_correlations.items():
    print(f"特征 {feature1} 与特征 {feature2} 的相关性：{correlation:.2f}")
