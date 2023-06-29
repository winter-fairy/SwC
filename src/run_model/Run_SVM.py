import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from src.utils.data_util import MyDataset

# 读取所有数据
all_data_path = '../../Data/processed/merged_data_KNN.csv'
all_data = pd.read_csv(all_data_path)

# 划分特征和标签
all_data_x = all_data.iloc[:, 1:-1]
all_data_y = all_data['label']

# 划分测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(all_data_x, all_data_y, test_size=0.2, random_state=42)


# 创建SVM分类器
svm_classifier = SVC(kernel='rbf')  # 可以根据需要选择不同的内核函数，如linear、rbf等

# 训练模型
svm_classifier.fit(x_train, y_train)

# 在测试集上进行预测
y_pred = svm_classifier.predict(x_test)

# 输出分类报告
print(classification_report(y_test, y_pred))
