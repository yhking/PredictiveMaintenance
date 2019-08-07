import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Activation
# from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

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
##测试数据中：获取每个ID设备的已经运行的周期值
rul = pd.DataFrame(pm_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'UL'] ##UL: Used Life 已经运行过的周期值
# rul.head()

##RUL+UL为设备正常运行的总生命周期值
pm_truth['TUL']=pm_truth['RUL']+rul['UL'] ##TUL: Total Useful Life 总寿命周期值
# pm_truth.head()

##给训练数据加入TTF标签，后面做训练时用（同ID划分为一组，获取TTF值）
pm_train['TTF'] = pm_train.groupby('id')['cycle'].transform(max)-pm_train['cycle']  ## TTF: Time To Failure, 距离发生故障的周期值
print("pm_train {0}\n{1}".format(pm_train.shape, pm_train.head()))

##给测试数据加入TTF标签，后面做算法验证用
pm_truth.drop('RUL', axis=1, inplace=True)
pm_test=pm_test.merge(pm_truth,on=['id'],how='left')
pm_test['TTF']=pm_test['TUL'] - pm_test['cycle']  ##TTF: Time To Failure, 距离发生故障的周期值
pm_test.drop('TUL', axis=1, inplace=True)
print("pm_test {0}\n{1}".format(pm_test.shape, pm_test.head()))


"""二分类模型"""
##为二分类模型准备训练和测试数据
df_train=pm_train.copy()
df_test=pm_test.copy()
period=30
df_train['label_bc'] = df_train['TTF'].apply(lambda x: 1 if x <= period else 0)
df_test['label_bc'] = df_test['TTF'].apply(lambda x: 1 if x <= period else 0)
# df_train.head()
# df_test.head()

features_col_name=['setting1', 'setting2', 'setting3', 
                   's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
                   's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
target_col_name='label_bc'


"""
数据可视化
"""
plt.plot(pm_train.groupby('id')['cycle'], pm_train.groupby('id'))

"""
归一化处理
"""
# sc=MinMaxScaler()
# df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])
# df_test[features_col_name]=sc.transform(df_test[features_col_name])









