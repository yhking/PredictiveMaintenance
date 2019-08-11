import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Activation
# from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')

##设置不换行显示DataFrame数据
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
"""
浏览数据
"""
##加载原始训练数据集（去掉无用的空列）
pm_train=pd.read_csv('./data/PM_train.txt',sep=' ',header=None).drop([26,27],axis=1)
col_names = ['id','cycle','setting1','setting2','setting3',
             's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
             's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
pm_train.columns=col_names
# print('Shape of Train dataset: ', pm_train.shape, type(pm_train))
# pm_train.head()

##加载原始测试数据集（去掉无用的空列）
pm_test=pd.read_csv('./data/PM_test.txt',sep=' ',header=None).drop([26,27],axis=1)
pm_test.columns=col_names
# print('Shape of Test dataset: ',pm_test.shape)
# pm_test.head()

##测试验证结果（去掉无用的空列）
pm_truth=pd.read_csv('./data/PM_truth.txt',sep=' ',header=None).drop([1],axis=1)
pm_truth.columns=['RUL'] ##RUL: Remaining Useful Life 剩余寿命周期
pm_truth['id']=pm_truth.index + 1
# pm_truth.head()

"""
清洗数据
"""
"""回归模型"""
##给训练数据加入TTF标签，后面做训练时用（同ID划分为一组，获取TTF值）
pm_train['TTF'] = pm_train.groupby('id')['cycle'].transform(max)-pm_train['cycle']  ## TTF: Time To Failure, 距离发生故障的周期值
train_columns = pm_train.columns

## 取id = 1引擎的数据
train_id_1 = pm_train[pm_train['id'] == 1]

train_id = train_id_1

## 归一化处理
# print("pm_train_1: {0}\n{1}".format(train_id.shape, train_id.head()))
norm_t = MinMaxScaler().fit_transform(train_id[train_columns[2:]]) ## array类型,MinMaxScaler默认按列执行归一化
norm_train = pd.DataFrame(norm_t, columns=train_columns[2:]) ## 转换为dataframe类型
# print("norm_train: {0}\n{1}".format(norm_train.shape, norm_train.head()))
norm_train = norm_train.join(train_id[['id','cycle']])
# print("norm_train: {0}\n{1}".format(norm_train.shape, norm_train.head()))
norm_train = norm_train[train_columns] ##调换字段顺序
# print("norm_train: {0}\n{1}".format(norm_train.shape, norm_train.head()))

"""
## 一张图显示所有
plt.figure(1)
ax1 = plt.subplot(5,1,1)
ax2 = plt.subplot(5,1,2)
ax3 = plt.subplot(5,1,3)
ax4 = plt.subplot(5,1,4)
ax5 = plt.subplot(5,1,5)
plt.sca(ax1) ## 选中ax1,绘图
sns.lineplot(data=norm_train[['setting1', 'setting2', 'setting3', 'TTF']])
plt.sca(ax2)
sns.lineplot(data=norm_train[norm_train.columns[5:11]])
plt.sca(ax3)
sns.lineplot(data=norm_train[norm_train.columns[11:17]])
plt.sca(ax4)
sns.lineplot(data=norm_train[norm_train.columns[17:23]])
plt.sca(ax5)
sns.lineplot(data=norm_train[norm_train.columns[23:26]])
plt.show()
"""

## 分开多张图显示
plt.figure(1)
sns.lineplot(data=norm_train[['setting1', 'setting2', 'setting3', 'TTF']])

plt.figure(2)
ax2 = plt.subplot(2,1,1)
ax3 = plt.subplot(2,1,2)
plt.sca(ax2)
sns.lineplot(data=norm_train[norm_train.columns[5:11]])
plt.sca(ax3)
sns.lineplot(data=norm_train[norm_train.columns[11:17]])

plt.figure(3)
ax4 = plt.subplot(2,1,1)
ax5 = plt.subplot(2,1,2)
plt.sca(ax4)
sns.lineplot(data=norm_train[norm_train.columns[17:23]])
plt.sca(ax5)
sns.lineplot(data=norm_train[norm_train.columns[23:26]])
plt.show()
