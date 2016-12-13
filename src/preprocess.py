#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

titanic_df = pd.read_excel('../titanic3.xls','titanic3',index_col=None,na_values=['NA'])
num1 = titanic_df['survived'].mean()
print("乘客总体生存几率：%f\n" % num1)
num2 = titanic_df.groupby('pclass').mean()
print("各阶级生存几率:")
print num2.to_string()
print "\n"

class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
print("各阶级性别 生存几率:")
print class_sex_grouping.to_string()
print "\n"
plt.figure('Class & Sex Factor')
plt.xlabel('Class & Sex')
plt.ylabel('Survived Rate')
plt.title('Class & Sex Factor')               
class_sex_grouping['survived'].plot.bar()

group_by_age = pd.cut(titanic_df['age'],np.arange(0,90,10))
age_grouping = titanic_df.groupby(group_by_age).mean()
print("各年龄段生存几率:")
print age_grouping.to_string()
print "\n"
plt.figure('Age Factor')
plt.title('Age Factor')
plt.xlabel('Age')
plt.ylabel('Survived Rate')
age_grouping['survived'].plot.bar(color='m')

print("未处理结果：")
print titanic_df.count()
print "\n"
titanic_df = titanic_df.drop(['body','cabin','boat'],axis=1)
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")
titanic_df = titanic_df.dropna()
print("处理后统计结果：")
print titanic_df.count()

#预处理数据
def preprocess_titanic_df(df):
	processed_df = df.copy()
	le = preprocessing.LabelEncoder()
	processed_df.sex = le.fit_transform(processed_df.sex)
	processed_df.embarked = le.fit_transform(processed_df.embarked)
	processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1)
	return processed_df
processed_df = preprocess_titanic_df(titanic_df)
processed_df.to_excel('data.xls')

plt.show()
