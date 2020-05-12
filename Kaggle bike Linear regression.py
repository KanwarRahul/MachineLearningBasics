# ----------------------------------------------
# Step 0 - Import Libraries
# ----------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
%matplotlib qt

# ----------------------------------------------
# Step 1 - Read the data
# ----------------------------------------------
bikes = pd.read_csv('hour.csv')


# ----------------------------------------------
# Step 2 - Prelim Analysis and Feature selection
# ----------------------------------------------
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)

bikes_prep.isnull().sum()

# Create pandas histogram
bikes_prep.hist(rwidth = 0.9)
plt.tight_layout()


plt.subplot(2,2,1)
plt.title('Temp vs demand')
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=2,c='g')


plt.subplot(2,2,2)
plt.title('aTemp vs demand')
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=2,c='b')


plt.subplot(2,2,3)
plt.title('Humidity vs demand')
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=2,c='m')

plt.subplot(2,2,4)
plt.title('Windspee vs demand')
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=2,c='c')


plt.tight_layout
colors=['g','r','m','b']
plt.subplot(3,3,1)
plt.title('average Demand Per season ')

cat_list=bikes_prep['season'].unique()

cat_average=bikes_prep.groupby('season').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)

plt.subplot(3,3,2)
plt.title('Average Demand per month')
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,3)
plt.title('Average Demand per Holiday')
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,4)
plt.title('Average Demand per Weekday')
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,5)
plt.title('Average Demand per Year')
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,6)
plt.title('Average Demand per hour')
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,7)
plt.title('Average Demand per Workingday')
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,8)
plt.title('Average Demand per Weather')
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.tight_layout()



plt.subplot(3,3,6)
plt.title('Average Demand per hour')
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

bikes_prep['demand
df3=np.log(df2)
plt.figure()
df2.hist(rwidth=0.9,bins=20)

plt.figure()
df3.hist(rwidth=0.9,bins=20)'].describe()

bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])


correlation=bikes_prep[['temp','atemp','humidity', 'windspeed','demand']].corr()

bikes_prep = bikes_prep.drop(['weekday', 'year', 'workingday', 'atemp', 'windspeed'], axis=1)

df1=pd.to_numeric(bikes_prep['demand'],downcast='float')

plt.acorr(df1,maxlags=12)


df2=bikes_prep['demand']

bikes_prep['demand']=np.log(bikes_prep['demand'])









