import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from src.utils.data_util import MyDataset
from src.models.MLP import MLP

# 读取数据，创建数据集类
file_path = '../../Data/train_10000.csv'
train_data = MyDataset(file_path)

# 设置超参数
input_size = 107  # 输入特征的维度
hidden_size = 64  # 隐藏层大小
output_size = 6  # 输出类别的数量
learning_rate = 0.0005
num_epochs = 70
batch_size = 32

# 初始化数据加载器
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# 初始化模型、损失函数
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # 交叉损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(features)  # 计算输出
        loss = criterion(outputs, labels)  # 计算损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss))

# 测试集数据
file_path = '../../Data/validate_1000.csv'
test_data = MyDataset(file_path)

# 测试集上评估
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))

