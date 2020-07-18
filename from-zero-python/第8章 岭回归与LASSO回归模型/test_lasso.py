
# #岭回归模型
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge,RidgeCV
import matplotlib.pyplot as plt

#读取数据
# 读取糖尿病数据集
diabetes = pd.read_excel(r'diabetes.xlsx', sep = '')
print(diabetes.head())

#构造自变量（剔除性别、年龄和因变量）
predictors=diabetes.columns[2:-1]
print(predictors)

#将数据集拆分为训练集和测试集
X_train,X_test,y_train,y_test=model_selection.train_test_split(
    diabetes[predictors],
    diabetes['Y'],
    test_size=0.2,
    random_state=1234
)

Lambdas=np.logspace(-5,2,200)
# # print(Lambdas)
# # print(len(Lambdas))
#
# #交叉验证，设置交叉验证的参数，对于每一个Lambdas值，都执行10重交叉验证
# ridge_cv=RidgeCV(alphas=Lambdas,normalize=True,scoring='neg_mean_squared_error',cv=10)
#
# #模型拟合
# ridge_cv.fit(X_train,y_train)
# ridge_best_Lambda=ridge_cv.alpha_
# print("最优的参数是: ",ridge_best_Lambda)
#
# ridge=Ridge(alpha=ridge_best_Lambda,normalize=True)
# ridge.fit(X_train,y_train)
#
# #返回岭回归系数
# res=pd.Series(index=['Intercept']+X_train.columns.tolist(),
#           data=[ridge.intercept_]+ridge.coef_.tolist())
# print(res)
#
# #模型预测
from sklearn.metrics import mean_squared_error
# ridge_predict=ridge.predict(X_test)
# RMSE=np.sqrt(mean_squared_error(y_test,ridge_predict))
# print(RMSE)




#LASSO回归模型
from sklearn.linear_model import Lasso,LassoCV

#LassoCV模型的交叉验证
lasso_cv=LassoCV(alphas=Lambdas,normalize=True,cv=10,max_iter=10000)
lasso_cv.fit(X_train,y_train)
lasso_best=lasso_cv.alpha_
print("最优解： ",lasso_best)

#建模
lasso=Lasso(alpha=lasso_best,normalize=True,max_iter=10000)
lasso.fit(X_train,y_train)
res=pd.Series(index=['Intercept']+X_train.columns.tolist(),
              data=[lasso.intercept_]+lasso.coef_.tolist())
print(res)

#模型评估
#模型预测
lasso_predict=lasso.predict(X_test)
RMSE=np.sqrt(mean_squared_error(y_test,lasso_predict))
print(RMSE)



























