# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # income=pd.read_csv('Salary_Data.csv')
# # print(income)
# # sns.lmplot(x='YearsExperience',y='Salary',data=income,ci=50)
# # plt.show()
# #
# import statsmodels.api as sm
# # fit=sm.formula.ols('Salary~YearsExperience',data=income).fit()
# # print(fit.params)
#
# from sklearn import model_selection

# Profit=pd.read_excel('Predict to Profit.xlsx')
# print(Profit)
#
# train,test=model_selection.train_test_split(Profit,test_size=0.2,random_state=1234)
# #根据train数据建模
# model=sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + C(State)',data=train).fit()
# print("回归参数: \n",model.params)
#
# test_x=test.drop(labels='Profit',axis=1)
# pred=model.predict(exog=test_x)
# print("预测与实际的差异: \n",pd.DataFrame({'Prediction':pred,'Real':test.Profit}))
# print("F: \n",model.fvalue)
# from scipy.stats import f
# p=model.df_model
# print("统计变量个数p： ",p)
# n=train.shape[0]
# print("观测个数n: ",n)
#
# F_Theroy=f.ppf(q=0.95,dfn=p,dfd=n-p-1)
# print("F： ",F_Theroy)
# print("t检验值\n ")
# print(model.summary())

































