# 项目配置
```angular2html
python == 3.7
torch == 1.13.1
scikit-learn == 1.0.2
numpy == 1.21.5
pandas == 1.3.5
seaborn == 0.12.2
```

# 数据解释
```angular2html
train_10000.csv 一万条数据，当训练集，有空值，有重复数据
validate_1000.csv 1000条数据，验证集数据，当测试集用
```

# 文档分类
### utils
```angular2html
data_util.py 用来进行数据处理的工具类，只需传入路劲计科
corr_analysis 对特征进行相关性分析，根据最终结果看来，此路不通
```
### models
```angular2html
MLP.py 多层感知机
SVM.py 软间隔SVM，线性核高斯核都用了
```
### run_models
```angular2html
Run_MLP.py 运行MLP的结果
Run_SVM.py 运行SVM的结果
```

# 项目进展
### 当前结果
```angular2html
1.MLP平均结果为78%左右
2.SVM平均结构为79%左右，高斯核79%，线性核75%，其他核都很垃圾
```
### 未来工作
```angular2html
1.未来一片迷茫
```

