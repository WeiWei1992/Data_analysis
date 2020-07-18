import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection

sports=pd.read_csv('Run or Walk.csv')
print(sports)

#提取所有自变量名称
predictors=sports.columns[4:]

#构建自变量矩阵
X=sports.loc[:,predictors]
print(X)

#提取y变量值
y=sports['activity']
print(y)

#将数据集拆分训练集和测试集
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.25,random_state=1234)

#利用训练集建模
sklearn_logistic=linear_model.LogisticRegression()
sklearn_logistic.fit(X_train,y_train)

#输出各个模型的参数
print(sklearn_logistic.intercept_,sklearn_logistic.coef_)

#模型预测
sklearn_predict=sklearn_logistic.predict(X_test)

#预测结果
print(pd.Series(sklearn_predict).value_counts())

#模型评估
from sklearn import metrics

#混肴矩阵
cm=metrics.confusion_matrix(y_test,sklearn_predict,labels=[0,1])
print(cm)

#准确率
Accuracy=metrics._scorer.accuracy_score(y_test,sklearn_predict)

print(Accuracy)

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.heatmap(cm,annot=True,fmt='.2e',cmap='GnBu')
# plt.show()

import matplotlib.pyplot as plt
#predict_proba得到的是正负的概率
res=sklearn_logistic.predict_proba(X_test)
print("res: ",res)
y_score=sklearn_logistic.predict_proba(X_test)[:,1]
print("y_score: ",y_score)
print("len(y_score): ",len(y_score))
fpr,tpr,threshold=metrics.roc_curve(y_test,y_score)
print("fpr: ",fpr)
print("len(fpr): ",len(fpr))
print("tpr: ",tpr)
print("len(tpr): ",len(tpr))
print("threshold: ",threshold)
print("len(threshold): ",len(threshold))

#计算AUC的值
roc_auc=metrics.auc(fpr,tpr)
print("roc_auc: ",roc_auc)

#绘制面积图
plt.stackplot(fpr,tpr,colors='steelblue',alpha=0.5,edgecolor='black')
#添加轮廓
plt.plot(fpr,tpr,color='black',lw=1)

#添加对角线
plt.plot([0,1],[0,1],color='red',linestyle='--')

#添加文本信息
plt.text(0.5,0.3,'ROC curve (area = %0.2f )'%roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()


