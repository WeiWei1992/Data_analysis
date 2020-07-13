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
edu = [0.2515,0.3724,0.3336,0.0368,0.0057]
labels = ['中专','大专','本科','硕士','其他']

explode = [0,0.1,0,0,0]  # 用于突出显示大专学历人群
colors=['#9999ff','#ff9999','#7777aa','#2442aa','#dd5555'] # 自定义颜色

# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 将横、纵坐标轴标准化处理，保证饼图是一个正圆，否则为椭圆
plt.axes(aspect='equal')

# 控制x轴和y轴的范围
plt.xlim(0,4)
plt.ylim(0,4)

plt.pie(
    x=edu,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%.1f%%',
    pctdistance=0.8,
    labeldistance=1.15,
    startangle=180,
    radius=1.5,
    counterclock=False,
    wedgeprops={'linewidth':1.5,'edgecolor':'green'},
    textprops={'fontsize':12,'color':'k'},
    center=(1.8,1.8),
    frame=1
)

plt.show()