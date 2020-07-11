
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



#练习2
import pandas as pd
path2='./pandas_exercise/exercise_data/Euro2012_stats.csv'
euro12=pd.read_csv(path2)
print(euro12.head())

#只选取某一列
print(euro12.Goals)
print(euro12['Goals'])

#获取行数
print(euro12.shape[0])

#试试一下git
#试一下git add

#获取列
print(euro12.shape[1])
print("xxx")

print("this is my pc")


