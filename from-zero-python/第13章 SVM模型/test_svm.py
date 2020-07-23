from sklearn import svm
import pandas as pd
from sklearn import model_selection
from sklearn import metrics

letters=pd.read_csv('letterdata.csv')
print(letters.head())
letters.head()

# 将数据拆分为训练集和测试集
predictors = letters.columns[1:]
X_train, X_test, y_train, y_test = model_selection.train_test_split(letters[predictors], letters.letter,
                                                                    test_size=0.25, random_state=1234)

# # 使用网格搜索法，选择线性可分SVM“类”中的最佳C值
# C = [0.1, 0.5]
# parameters = {'C': C}
# grid_linear_svc = model_selection.GridSearchCV(estimator=svm.LinearSVC(), param_grid=parameters, scoring='accuracy',
#                                                cv=5, verbose=1)
# # 模型在训练数据集上的拟合
# grid_linear_svc.fit(X_train, y_train)
# # 返回交叉验证后的最佳参数值
#
# print(grid_linear_svc.best_params_,grid_linear_svc.best_score_)
# pred_linear_svc=grid_linear_svc.predict(X_test)
# res=metrics.accuracy_score(y_test,pred_linear_svc)
# print("准去率：\n",res)

kernel=['rbf','sigmoid']
C=[5]
parameters={'kernel':kernel,'C':C}
grid_svc=model_selection.GridSearchCV(estimator=svm.SVC(),
                                      param_grid=parameters,
                                      scoring='accuracy',
                                      cv=2,
                                      verbose=1)
#拟合
grid_svc.fit(X_train,y_train)

print("最佳参数")
print(grid_svc.best_params_,grid_svc.best_score_)

pred_svc=grid_svc.predict(X_test)
res=metrics.accuracy_score(y_test,pred_svc)
print("准去率： \n",res)
