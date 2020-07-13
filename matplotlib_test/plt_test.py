import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# # GDP = [12406.8,13908.57,9386.87,9143.64]
# #
# # plt.barh(range(4),GDP,align='center',color='steelblue',alpha=0.8)
# # plt.ylabel('GDP')
# # plt.title('this is title')
# # plt.xticks(range(4),['北京','上海','天津','重庆'])
# # plt.ylim([5000,15000])
#
# # for x,y in enumerate(GDP):
# #     print(x,y)
# #     plt.text(x,y+100,'%s'%round(y,1),ha='center')
#
#
# # 构建数据
# Y2016 = [15600,12700,11300,4270,3620]
# Y2017 = [17400,14800,12000,5200,4020]
# labels = ['北京','上海','香港','深圳','广州']
# bar_width = 0.35
#
# plt.bar(np.arange(5),Y2016,label='2016',color='steelblue',alpha=0.8,width=bar_width)
# plt.bar(np.arange(5)+bar_width,Y2017,label='2017',color='indianred',alpha=0.8,width=bar_width)
#
# plt.xlabel('Top5城市')
# plt.ylabel('家庭数量')
# plt.title('亿万财富家庭数Top5城市分布')
# plt.legend()
# for x2016,y2016 in enumerate(Y2016):
#     print(x2016,y2016)
#     plt.text(x2016,y2016+100,'%s' %y2016,ha='center')
# for x2017,y2017 in enumerate(Y2017):
#     print(x2017,y2017)
#     plt.text(x2017+bar_width,y2017+100,'%s'%y2017,ha='center')
# plt.show()


#绘制饼图
# edu = [0.2515,0.3724,0.3336,0.0368,0.0057]
# labels = ['中专','大专','本科','硕士','其他']
#
# explode = [0,0.1,0,0,0]  # 用于突出显示大专学历人群
# colors=['#9999ff','#ff9999','#7777aa','#2442aa','#dd5555'] # 自定义颜色
#
# # 中文乱码和坐标轴负号的处理
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 将横、纵坐标轴标准化处理，保证饼图是一个正圆，否则为椭圆
# plt.axes(aspect='equal')
#
# # 控制x轴和y轴的范围
# plt.xlim(0,4)
# plt.ylim(0,4)
#
# plt.pie(
#     x=edu,
#     explode=explode,
#     labels=labels,
#     colors=colors,
#     autopct='%.1f%%',
#     pctdistance=0.8,
#     labeldistance=1.15,
#     startangle=180,
#     radius=1.5,
#     counterclock=False,
#     wedgeprops={'linewidth':1.5,'edgecolor':'green'},
#     textprops={'fontsize':12,'color':'k'},
#     center=(1.8,1.8),
#     frame=1
# )
#
# plt.show()

#绘制箱线图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
path='../pandas_test/pandas_exercise/exercise_data/train.csv'
titanic=pd.read_csv(path)
print(titanic)
titanic.dropna(subset=['Age'],inplace=True)
print(titanic)

# # 设置中文和负号正常显示
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.rcParams['axes.unicode_minus'] = False
#
# # plt.boxplot(x=titanic.Age,
# #             patch_artist=True,
# #             showmeans=True,
# #             boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
# #             flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
# #             meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色
# #             medianprops={'linestyle': '--', 'color': 'orange'}  # 设置中位数线的属性，线的类型和颜色
# #             )
# #
# # # 设置y轴的范围
# # plt.ylim(0,85)
# #
# # # 去除箱线图的上边框与右边框的刻度标签
# # plt.tick_params(top='off', right='off')
# # # 显示图形
# #
# # plt.show()
#
# titanic.sort_values(by='Pclass',inplace=True)
# print(titanic)
# Age=[]
# Levels=titanic.Pclass.unique()
# print(Levels)
# for Pclass in Levels:
#     Age.append(titanic.loc[titanic.Pclass==Pclass,'Age'])
# print("xxxxxxxxxxxx")
# print(Age)
# print(type(Age))
# print(len(Age))
#
# plt.boxplot(
#     x=Age,
#     patch_artist=True,
#     labels=['一等','二等','三等'],
#     boxprops = {'color':'black','facecolor':'#9999ff'},
#     flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
#     meanprops = {'marker':'D','markerfacecolor':'indianred'},
#     medianprops = {'linestyle':'--','color':'orange'}
# )
#
# plt.show()

# plt.hist(
#     titanic.Age,
#     bins=20,
#     color='steelblue',
#     edgecolor='k',
#     label='直方图'
# )
# # plt.tick_params(top='off',right='off')
# plt.legend()
# plt.show()

# plt.hist(
#     # titanic.Age,
#     # bins=np.arange(titanic.Age.min(),titanic.Age.max(),5),
#     # #normed=True,
#     # cumulative=True,
#     # color='steelblue',
#     # edgecolor='k',
#     # label='直方图'
#     titanic.Age,
#     bins = np.arange(titanic.Age.min(),titanic.Age.max(),5), # 指定直方图的组距
#     normed = True, # 设置为频率直方图
#     color = 'steelblue', # 指定填充色
#     edgecolor = 'k'     # 指定直方图的边界色
# )
#
# plt.title('乘客年龄累计频率直方图')
# plt.xlabel('年龄')
# plt.ylabel('累计频率')
# plt.legend(loc='best')
# plt.show()


age_female=titanic.Age[titanic.Sex=='female']
age_male=titanic.Age[titanic.Sex=='male']
# 设置直方图的组距
bins = np.arange(titanic.Age.min(), titanic.Age.max(), 2)
# 男性乘客年龄直方图
plt.hist(age_male, bins = bins, label = '男性', color = 'steelblue', alpha = 0.7)
# 女性乘客年龄直方图
plt.hist(age_female, bins = bins, label = '女性', color='r',alpha = 0.6)

# 设置坐标轴标签和标题
plt.title('乘客年龄直方图')
plt.xlabel('年龄')
plt.ylabel('人数')

# 显示图例
plt.legend()
# 显示图形
plt.show()







