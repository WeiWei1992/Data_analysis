
import pandas as pd
# skin=pd.read_excel('Skin_Segment.xlsx')
# print(skin)
# skin.y=skin.y.map({2:0,1:1})
# print(skin)
# print(skin.y.value_counts())
#
# from sklearn import model_selection
# X_train,X_test,y_train,y_test=model_selection.train_test_split(skin.iloc[:,:3],skin.y,
#                                                                test_size=0.25,random_state=1234)
#
# from sklearn import naive_bayes
# gnb=naive_bayes.GaussianNB()
# gnb.fit(X_train,y_train)
#
# #预测
# gnb_pred=gnb.predict(X_test)
# print(pd.Series(gnb_pred).value_counts())
#
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # #构建混肴矩阵
# # cm=pd.crosstab(gnb_pred,y_test)
# # #绘制混肴矩阵
# # sns.heatmap(cm,annot=True,cmap='GnBu',fmt='d')
# # plt.xlabel('Real')
# # plt.ylabel('Predict')
# # plt.show()
#
# print("模型的准确率为：\n",metrics.accuracy_score(y_test,gnb_pred))
# print("模型的评估报告\n",metrics.classification_report(y_test,gnb_pred))

# print("离散数据")
# mushrooms=pd.read_csv("mushrooms.csv")
# print(mushrooms.head())
#
# columns=mushrooms.columns[1:]
# print(columns)
# for column in columns:
#     mushrooms[column]=pd.factorize(mushrooms[column])[0]
#     res=pd.factorize(mushrooms[column])
#     print("res\n",res)
#     print("len(res)\n",len(res))
#
# print(mushrooms.head())
#
# from sklearn import model_selection
#
# Predictors=mushrooms.columns[1:]
# X_train,X_test,y_train,y_test=model_selection.train_test_split(mushrooms[Predictors],
#                                                                mushrooms['type'],
#                                                                test_size=0.25,
#                                                                random_state=10)
# #构建多项式分类
# from sklearn import naive_bayes
# mnb=naive_bayes.MultinomialNB()
# #拟合
# mnb.fit(X_train,y_train)
# #预测
# mnb_pred=mnb.predict(X_test)
# # #构建混肴矩阵
# # cm=pd.crosstab(mnb_pred,y_test)
# # import seaborn as sns
# # sns.heatmap(cm,annot=True,cmap='GnBu',fmt='d')
# # import matplotlib.pyplot as plt
# # plt.show()
# from sklearn import metrics
# print("模型的准去率是： \n",metrics.accuracy_score(y_test,mnb_pred))

import pandas as pd
evaluation=pd.read_excel('Contents.xlsx',sheet_name=0)
print(evaluation.head(10))
print(evaluation.Content.head(10))
evaluation.Content=evaluation.Content.str.replace('[0-9a-zA-Z]','')
print(evaluation.head(10))
print(evaluation.Content.head(10))

import jieba
#加入自定义词库
jieba.load_userdict('all_words.txt')

#读入停止词
with open('mystopwords.txt',encoding='utf-8') as words:
    stop_words=[i.strip() for i in words.readlines()]

#构造切词的自定义函数，并在切词过程中删除停止词
def cut_word(sentence):
    words=[i for i in jieba.lcut(sentence) if i not in stop_words]
    #切完词用空格隔开
    result=' '.join(words)
    return result
words=evaluation.Content.apply(cut_word)
print(words)

from sklearn.feature_extraction.text import CountVectorizer
counts=CountVectorizer(min_df=0.01)
print("counts: \n",counts)
#文档词条矩阵
dtm_counts=counts.fit_transform(words).toarray()
print("dtm_counts: \n",dtm_counts)
columns=counts.get_feature_names()
print("columns: \n",columns)

X=pd.DataFrame(dtm_counts,columns=columns)
y=evaluation.Type
print(X.head())

