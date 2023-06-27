import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from src.utils.data_util import MyDataset

# 读取训练数据
train_data_path = '../../Data/raw/train_10000.csv'
train_data = MyDataset.process_data(train_data_path)
# 分割特征和标签
train_features = train_data.iloc[:, 1:-1]  # 选择feature0到feature106列作为特征
train_labels = train_data['label']

# 读取测试集数据
test_data_path = '../../Data/raw/validate_1000.csv'
test_data = MyDataset.process_data(test_data_path)
# 分割特征和标签
test_features = test_data.iloc[:, 1:-1]  # 选择feature0到feature106列作为特征
test_labels = test_data['label']

# 创建SVM分类器
svm_classifier = SVC(kernel='linear')  # 可以根据需要选择不同的内核函数，如linear、rbf等

# 训练模型
svm_classifier.fit(train_features, train_labels)

# 在测试集上进行预测
test_pred = svm_classifier.predict(test_features)

# 输出分类报告
print(classification_report(test_labels, test_pred))
