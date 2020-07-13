
# #练习1
# import pandas as pd
# path1='./pandas_exercise/exercise_data/chipotle.tsv'
# chipo=pd.read_csv(path1,sep='\t')
# print(chipo.head(10))
#
# #获取数据列数
# print(chipo.shape[1])
#
# #获取全部列的名称
# print(chipo.columns)
# #行索引
# print(chipo.index)
#
# c=chipo[['item_name','quantity']].groupby(['item_name'],as_index=False).agg({'quantity':sum})
# b=c.sort_values(['quantity'],ascending=False)
# print(b.head())
#
# print(chipo['item_name'].nunique())
# print(chipo['choice_description'].value_counts().head())
# total_items_orders=chipo['quantity'].sum()
# print(total_items_orders)
#
# dollarizer=lambda x:float(x[1:-1])
# chipo['item_price']=chipo['item_price'].apply(dollarizer)
# print(chipo.head())
#
# chipo['sub_total']=round(chipo['item_price']*chipo['quantity'],2)
# sum=chipo['sub_total'].sum()
# print(sum)
# print(chipo.head())
#
# print(chipo['order_id'].nunique())



# #练习2
# import pandas as pd
# path2='./pandas_exercise/exercise_data/Euro2012_stats.csv'
# euro12=pd.read_csv(path2)
# print(euro12.head())
#
# #只选取某一列
# print(euro12.Goals)
# print(euro12['Goals'])
#
# #获取行数
# print(euro12.shape[0])
#
# #试试一下git
# #试一下git add
#
# #获取列
# print(euro12.shape[1])
#
# discipline=euro12[['Team','Yellow Cards','Red Cards']]
# print(discipline)
# x=discipline.sort_values(['Red Cards','Yellow Cards'],ascending=False)
# print(discipline)
# print(round(discipline['Yellow Cards'].mean()))
# print(euro12[euro12.Goals>6])
# print(euro12[euro12.Team.str.startswith('G')])
#
# print(euro12.iloc[:,0:7])
# print(euro12.iloc[:,:-3])
#
# print(euro12.loc[euro12.Team.isin(['England','Italy','Russia']),['Team','Shooting Accuracy']])


# #练习3 数据分组
# import pandas as pd
#
# path3='./pandas_exercise/exercise_data/drinks.csv'
# drinks=pd.read_csv(path3)
# print(drinks.head())
# print(drinks.groupby('continent').beer_servings.mean())
# print(drinks.groupby('continent').wine_servings.describe())
# print(drinks.groupby('continent').median())
# print(drinks.groupby('continent').beer_servings.agg(['mean','min','max']))


##练习4 Apply函数
# import numpy as np
# import pandas as pd
#
# path4='./pandas_exercise/exercise_data/US_Crime_Rates_1960_2014.csv'
# crime=pd.read_csv(path4)
# print(crime.head())
#
# print(crime.info())
# crime.Year=pd.to_datetime(crime.Year,format='%Y')
# print(crime.info())
#
# crime=crime.set_index('Year',drop=True)
# print(crime.head())
#
# del crime['Total']
# print(crime.head())
#
# print(crime.idxmax(0))

# #练习5 合并
# import numpy as np
# import pandas as pd
# # path5='./pandas_exercise/exercise_data/US_Crime_Rates_1960_2014.csv'
#
# raw_data_1 = {
#         'subject_id': ['1', '2', '3', '4', '5'],
#         'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
#         'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
#
# raw_data_2 = {
#         'subject_id': ['4', '5', '6', '7', '8'],
#         'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
#         'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
#
# raw_data_3 = {
#         'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
#         'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
# data1=pd.DataFrame(raw_data_1,columns=['subject_id','first_name','last_name'])
# data2=pd.DataFrame(raw_data_2,columns=['subject_id','first_name','last_name'])
# data3=pd.DataFrame(raw_data_3,columns=['subject_id','test_id'])
# print(data1)
# print(data2)
#
# all_data=pd.concat([data1,data2])
# print(all_data)
# print("xxxxxxx")
# all_data_col=pd.concat([data1,data2],axis=1)
# print(all_data_col)
# print(data3)
# print(pd.merge(all_data,data3,on='subject_id'))
# print("=======================")
# print(pd.merge(data1,data2,on='subject_id',how='inner'))
# print(pd.merge(data1,data2,on='subject_id',how='outer'))


# #练习6统计
# import pandas as pd
# import datetime
# path6='./pandas_exercise/exercise_data/wind.data'
# data1=pd.read_table(path6,sep='\s+')
# print(data1)
# data=pd.read_table(path6,sep='\s+',parse_dates=[[0,1,2]])
# print(data.head())
#
# def fix_century(x):
#     year=x.year-100 if x.year>1989 else x.year
#     return datetime.date(year,x.month,x.day)
# data['Yr_Mo_Dy']=data['Yr_Mo_Dy'].apply(fix_century)
# print(data.head())
#
# data["Yr_Mo_Dy"]=pd.to_datetime(data["Yr_Mo_Dy"])
# data=data.set_index('Yr_Mo_Dy')
# print(data.head())
# print(data.isnull().sum())
# print(data.shape[0]-data.isnull().sum())
#
#
# print(data.mean().mean())
#
# print("-------------")
# print(data.mean(axis=0))
#
# loc_stats=pd.DataFrame()
# loc_stats['min']=data.min()
# loc_stats['max']=data.max()
# loc_stats['mean']=data.mean()
# loc_stats['std']=data.std()
# print(loc_stats)
# print(data.head)
# day_stats = pd.DataFrame()
#
# # this time we determine axis equals to one so it gets each row.
# day_stats['min'] = data.min(axis = 1) # min
# day_stats['max'] = data.max(axis = 1) # max
# day_stats['mean'] = data.mean(axis = 1) # mean
# day_stats['std'] = data.std(axis = 1) # standard deviations
# print(day_stats)
#
# data['date'] = data.index
#
# # creates a column for each value from date
# data['month'] = data['date'].apply(lambda date: date.month)
# data['year'] = data['date'].apply(lambda date: date.year)
# data['day'] = data['date'].apply(lambda date: date.day)
# print(data)
#
# january_winds=data.query('month==1')
# print(january_winds)
# print(january_winds.loc[:,'RPT':'ROS'].mean())
#
# print(data.query('day==1'))

# #练习7 可视化
# path7='./pandas_exercise/exercise_data/train.csv'
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# titanic=pd.read_csv(path7)
# print(titanic.head())
# titanic.set_index('PassengerId').head()
# x=titanic.set_index('PassengerId')
# print(x.head())
#
# # males=(titanic['Sex']=='male').sum()
# # females=(titanic['Sex']=='female').sum()
# #
# # proportions=[males,females]
# # plt.pie(
# #
# #     proportions,
# #     labels=['Males','Females'],
# #     shadow=False,
# #     colors=['blue','red'],
# #     explode=(0.15,0),
# #     startangle=90,
# #     autopct='%1.1f%%'
# # )
# #
# # plt.axis('equal')
# # plt.title('Sex Proportion')
# # plt.tight_layout()
# # plt.show()
#
# # lm=sns.lmplot(x='Age',y='Fare',data=titanic,hue='Sex',fit_reg=False)
# # lm.set(title='Fare x Age')
# #
# # axes=lm.axes
# # axes[0,0].set_ylim(-5,)
# # axes[0,0].set_xlim(-5,85)
# #
# # plt.show()
#
# print(titanic.Survived.sum())
#
# df=titanic.Fare.sort_values(ascending=False)
# print(df)
#
# binsVal=np.arange(0,600,10)
# print(binsVal)
# plt.hist(df,bins=binsVal)
# plt.xlabel('Fare')
# plt.ylabel('Frequency')
# plt.title('Fare Payed Histrogram')

# plt.show()

# #练习8 创建数据框
# import pandas as pd
# raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
#             "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
#             "type": ['grass', 'fire', 'water', 'bug'],
#             "hp": [45, 39, 44, 45],
#             "pokedex": ['yes', 'no','yes','no']
#             }
# print(raw_data)
# pokemon=pd.DataFrame(raw_data)
# print(pokemon.head())
# pokemon=pokemon[['name','type','hp','evolution','pokedex']]
# print(pokemon)
#
# pokemon['place'] = ['park','street','lake','forest']
# print(pokemon)

#print(pokemon.dtypes)


# #练习9 时间序列
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# path9='./pandas_exercise/exercise_data/Apple_stock.csv'
# apple=pd.read_csv(path9)
# print(apple.head())
# print(apple.dtypes)
#
# apple.Date=pd.to_datetime(apple.Date)
# print(apple.Date.head())
# apple=apple.set_index('Date')
# print(apple.head())
# print(apple.index.is_unique)
# print(apple.sort_index(ascending=True).head())
# print((apple.index.max()-apple.index.min()).days)

#练习10 删除数据
path10='./pandas_exercise/exercise_data/iris.csv'
import pandas as pd
import numpy as np
iris=pd.read_csv(path10)
print(iris.head())

iris = pd.read_csv(path10,names = ['sepal_length','sepal_width', 'petal_length', 'petal_width', 'class'])
print(iris.head())
print(pd.isnull(iris).sum())
iris.iloc[10:20,2:3]=np.nan
print(iris.head(20))
print("xxxxxxxxxxxxx")
# iris.petal_length.fillna(1,inplace=True)
# print(iris.head(20))
iris.fillna(1,inplace=True)
print(iris.head(20))

# del iris['class']
xx=iris.drop('class',axis=1)
print(xx.head())

iris=iris.dropna(how='any')
print(iris.head())