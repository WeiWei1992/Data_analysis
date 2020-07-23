import pandas as pd
from sklearn import model_selection
Knowledge=pd.read_excel('Knowledge.xlsx')
print(Knowledge.head())

predictors = Knowledge.columns[:-1]
print(predictors)

X_train,X_test,y_train,y_test=model_selection.train_test_split(Knowledge[predictors],
                                                               Knowledge['UNS'],
                                                               test_size=0.25,
                                                               random_state=1234)

from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
#设置待测试的不同k值
K=np.arange(1,np.ceil(np.log2(Knowledge.shape[0])))
print(K)
print(Knowledge.shape[0])

#构建空的列表，用于存储平均准确率
accuracy=[]
for k in K:
    print("k: ",k)
    print("type(k): ",type(k))
    k=int(k)
    knn=neighbors.KNeighborsClassifier(n_neighbors=k,weights='distance')
    cv_result=model_selection.cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    #cv_result = model_selection.cross_val_score(neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance'),
    #                                            X_train, y_train, cv=10, scoring='accuracy')
    accuracy.append(cv_result.mean())

#从k个平均准确率中选出最大值所对应的下标
print("accuracy:\n",accuracy)
arg_max=np.array(accuracy).argmax()

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# plt.plot(K,accuracy)
# plt.scatter(K,accuracy)
#
# #添加文字说明
# plt.text(K[arg_max],accuracy[arg_max],'最佳k值为%s '%int(K[arg_max]))
# plt.show()

#重建模型
knn_class=neighbors.KNeighborsClassifier(n_neighbors=6,weights='distance')

#模型拟合
knn_class.fit(X_train,y_train)

#预测
predict=knn_class.predict(X_test)

#构建混肴矩阵
cm=pd.crosstab(predict,y_test)
print(cm)

# import seaborn as sns
# cm=pd.DataFrame(cm,columns=['High','Low','Middle','Very_Low'],index=['High','Low','Middle','Very Low'])
# sns.heatmap(cm,annot=True,cmap='GnBu')
# plt.xlabel('Real Label')
# plt.ylabel('Predict Label')
# plt.show()

from sklearn import metrics
res=metrics._scorer.accuracy_score(y_test,predict)
print("res:\n",res)
print(metrics.classification_report(y_test,predict))

print("===============knn预测===================")
ccpp=pd.read_excel('CCPP.xlsx')
print(ccpp.head())
print(ccpp.shape)

from sklearn.preprocessing import minmax_scale

#对数据进行标准化处理
predictors=ccpp.columns[:-1]
X=minmax_scale(ccpp[predictors])

#设置待测试的不同k值
K=np.arange(1,np.log2(ccpp.shape[0]))
print("K:\n",K)
mse=[]
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, ccpp.PE,
                                                                    test_size = 0.25, random_state = 1234)

# for k in K:
#     k=int(k)
#     knn=neighbors.KNeighborsRegressor(n_neighbors=k,weights='distance')
#     cv_result=model_selection.cross_val_score(knn,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
#     mse.append((-1 * cv_result).mean())
#
# arg_min=np.array(mse).argmin()
# plt.plot(K,mse)
# plt.scatter(K,mse)
# plt.text(K[arg_min],mse[arg_min]+0.5,'最佳k值为%s' %int(K[arg_min]))
# plt.show()

knn_reg=neighbors.KNeighborsRegressor(n_neighbors=7,weights='distance')

#模拟拟合
knn_reg.fit(X_train,y_train)
#预测
predict=knn_reg.predict(X_test)
#计算mse
res=metrics.mean_squared_error(y_test,predict)
print("res:\n",res)

resfram=pd.DataFrame({'Real':y_test,'Predict':predict},columns={'Real','Predict'}).head(10)
print(resfram)







