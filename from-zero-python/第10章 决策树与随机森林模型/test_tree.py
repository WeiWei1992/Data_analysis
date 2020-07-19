'''
随机森林解决分类问题
'''
'''
import pandas as pd

Titanic=pd.read_csv('Titanic.csv')
print(Titanic.head())

Titanic.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
print(Titanic.head())
print(Titanic.isnull().sum(axis=0))

print(Titanic)

Titanic=Titanic.dropna(axis=0,how='any')
print(Titanic)

Titanic.Pclass=Titanic.Pclass.astype('category')
dummy=pd.get_dummies(Titanic[['Sex','Embarked','Pclass']])
Titanic=pd.concat([Titanic,dummy],axis=1)
print(Titanic.head())
Titanic.drop(['Sex','Embarked','Pclass'],inplace=True,axis=1)
print(Titanic.head())

from sklearn import model_selection

predictors=Titanic.columns[1:]
X_train,X_test,y_train,y_test=model_selection.train_test_split(Titanic[predictors],Titanic.Survived,test_size=0.25,random_state=1234)

print("xxx")

from sklearn.model_selection import GridSearchCV
from sklearn import tree

max_depth=[2,3,4,5,6]
min_samples_split=[2,4,6,8]
min_samples_leaf=[2,4,8,10,12]
parameters={'max_depth':max_depth,'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf}

#网络搜索法，测试不同的参数值
grid_dtcateg=GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                          param_grid=parameters,cv=10)

grid_dtcateg.fit(X_train,y_train)


# max_depth = [2,3,4,5,6]
# min_samples_split = [2,4,6,8]
# min_samples_leaf = [2,4,8,10,12]
# # 将各参数值以字典形式组织起来
# parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
# # 网格搜索法，测试不同的参数值
# grid_dtcateg = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = parameters, cv=10)
# # 模型拟合
# grid_dtcateg.fit(X_train, y_train)

#返回最佳参数
print(grid_dtcateg.best_params_)

#构建决策树
CART_Class=tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=4,min_samples_split=2)
#模型拟合
decision_tree=CART_Class.fit(X_train,y_train)

#预测
pred=CART_Class.predict(X_test)

# CART_Class = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf = 4, min_samples_split=2)
# # 模型拟合
# decision_tree = CART_Class.fit(X_train, y_train)
# # 模型在测试集上的预测
# pred = CART_Class.predict(X_test)

from sklearn import metrics
print("模型的准确率：",metrics.accuracy_score(y_test,pred))
#模型的准确率： 0.797752808988764
#0.797752808988764

import matplotlib.pyplot as plt
y_score=CART_Class.predict_proba(X_test)[:,1]
fpr,tpr,threshold=metrics.roc_curve(y_test,y_score)

#计算AUC的值
roc_auc=metrics.auc(fpr,tpr)
print("roc_auc: ",roc_auc)

# plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
# plt.plot(fpr,tpr,color='black',lw=1)
# plt.plot([0,1],[0,1],color='red',linestyle='--')
# plt.text(0.5,0.3,'ROC curve (area = %0.2f)' %roc_auc)
# plt.xlabel('1-Specificity')
# plt.ylabel('Sensitivity')
# plt.show()


# #画树
# from sklearn.tree import export_graphviz
# from IPython.display import Image
# #import pydotplus
# import pydot
# features_list=list(X_train.columns)
# export_graphviz(
#     decision_tree,
#     out_file='tree.dot',
#     feature_names=features_list,
#     filled=True,
#     rounded=True,
#     special_characters=True
# )
# (graph,)=pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')

print("开始随机森林模型")
from sklearn import ensemble
RF_class=ensemble.RandomForestClassifier(n_estimators=200,random_state=1234)

#随机森林的拟合
RF_class.fit(X_train,y_train)

#模型预测
RFclass_pred=RF_class.predict(X_test)
print("模型在测试集上的准确率：\n",metrics.accuracy_score(y_test,RFclass_pred))

#计算绘图数据
y_score=RF_class.predict_proba(X_test)[:,1]
fpr,tpr,threshold=metrics.roc_curve(y_test,y_score)
roc_auc=metrics.auc(fpr,tpr)

# #h绘图ROC曲线
# plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
# plt.plot(fpr,tpr,color='black',lw=1)
# plt.plot([0,1],[0,1],color='red',linestyle='--')
# plt.text(0.5,0.3,'ROC curve (area = %0.2f )' %roc_auc)
# plt.xlabel('1-Specificity')
# plt.ylabel('Sensitivity')
# plt.show()


#计算因变量的重要程度
importance=RF_class.feature_importances_
print("importance\n ",importance)
Impt_Series=pd.Series(importance,index=X_train.columns)
print("Impt_Series\n",Impt_Series)
Impt_Series.sort_values(ascending=True).plot(kind='barh')
plt.show()

'''

###随机森林解决预测问题
import pandas as pd
import xlrd
from sklearn import model_selection
NHANES=pd.read_excel('NHANES.xlsx')
print(NHANES.head())

predictors=NHANES.columns[:-1]
print("predictors: ",predictors)

X_train,X_test,y_train,y_test=model_selection.train_test_split(NHANES[predictors],
                                                               NHANES.CKD_epi_eGFR,
                                                               train_size=0.25,
                                                               random_state=1234)

#预设个参数
max_depth=[18,19,20,21,22]
min_samples_split=[2,4,6,8]
min_samples_leaf=[2,4,8,10,12]
parameters={'max_depth':max_depth,'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf}

# #网格搜索法，测试不同的参数值
# from sklearn.model_selection import GridSearchCV
from sklearn import tree
# grid_dtcateg=GridSearchCV(estimator=tree.DecisionTreeRegressor(),param_grid=parameters,cv=10)
#
# #模型拟合
# grid_dtcateg.fit(X_train,y_train)
# print("最佳参数 \n",grid_dtcateg.best_params_)

# #构建回归的决策树
# CART_Reg=tree.DecisionTreeRegressor(max_depth=20,min_samples_leaf=2,
#                                     min_samples_split=4)
# #拟合
# CART_Reg.fit(X_train,y_train)
# #预测
# pred=CART_Reg.predict(X_test)
#
# #计算MSE
# from sklearn import metrics
# mse=metrics.mean_squared_error(y_test,pred)
# print("MSE:\n",mse)
# # 3.537947265776475

#构建回归森林
from sklearn import ensemble
RF=ensemble.RandomForestRegressor(n_estimators=200,random_state=1234)
#拟合
RF.fit(X_train,y_train)
#预测
RF_pred=RF.predict(X_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,RF_pred)
print("mes\n",mse)

#绘制重要性图
import matplotlib.pyplot as plt
importance=pd.Series(RF.feature_importances_,index=X_train.columns)
importance.sort_values().plot(kind='barh')
plt.show()




