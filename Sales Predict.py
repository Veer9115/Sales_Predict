# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:12:03 2018

@author: pranv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data_Sales.csv')
df1 = pd.read_csv('Data_Sales.csv')


df.head()
df.describe()
df.isna().sum()

df = df.replace(0, np.nan)
df['Item_Fat_Content'].value_counts()

#To see unique values in the dataset
df.apply(lambda x: len(x.unique()))

#We see that LF, reg, low fat, are other names for Low Fat and Regular. We will combine these values.
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})

def plott(column):
    plt.bar(df[column].unique(), df[column].value_counts(), align='center')
    plt.ylabel('Value')
    plt.xlabel(column)
    plt.title('Value Counts for ' + column)
    plt.show()


df['Item_Type'].value_counts()

df['Outlet_Size']= df['Outlet_Size'].replace({np.nan: 'High'})

df['Outlet_Establishment_Year'] = 2013 - df['Outlet_Establishment_Year']
df = df.rename(columns = {'Outlet_Establishment_Year': 'Years_In_Business'})

plott('Item_Fat_Content')
plott('Outlet_Size')
plott('Outlet_Location_Type')
plott('Outlet_Type')


#Imputing values for Item_Weight using the Item_Identifier
item_avg_weight = df.pivot_table(values='Item_Weight', index='Item_Identifier')
miss_bool = df['Item_Weight'].isnull()
df.loc[miss_bool,'Item_Weight']  = df.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.at[x,'Item_Weight'])

#Imputing values for Item_Visibility using the Item_Identifier
item_avg_vis = df.pivot_table(values='Item_Visibility', index='Item_Identifier')
miss_bool1 = df['Item_Visibility'].isnull()
df.loc[miss_bool1,'Item_Visibility']  = df.loc[miss_bool1,'Item_Identifier'].apply(lambda x: item_avg_vis.at[x,'Item_Visibility'])

df.isna().sum()

df.dropna(how = 'any', inplace = True)

df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
df['Item_Type_Combined'] = df['Item_Type_Combined'].replace({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
df.drop('Item_Type',axis = 1, inplace = True)

#Creating Dummy Variables
df1 = pd.get_dummies(df, columns = ['Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Combined'], drop_first = True)

#Creating dataframe for predictor variables
X1 = df1.iloc[:, 1:5]
X2 = df1.iloc[:, 6:]

#Combining predictor variables
X = pd.concat([X1, X2], axis = 1).values
y = df1.iloc[:, 5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling - Done to ease calculations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Fitting the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Model Check
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean() #We get an accuracy of 0.52

#RMSE 
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred)) #RMSE score 1180



