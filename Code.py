#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 00:32:18 2021

@author: kwanw4
"""
#pip install numpy_financial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import numpy_financial as npf



# SECTION 1 READ FILES & DATA CLEANSING 
#-----------------------------------------------------------------------------

#Read Class A Housing Prices
df1=pd.read_csv("/Users/kwanw4/Downloads/FINA2390_Project 3/average_prices_by_class.csv")
df1.drop(df1.head(9).index,inplace=True)
df1.drop(df1.tail(8).index,inplace=True)
df1=df1[df1.columns[1:15]]
df1[df1.columns[0]]=df1[df1.columns[0]].fillna(method="ffill")
df1=df1.dropna(axis=1, how='any')
df1.columns =['Year', 'Month','Class_A_HK','Class_A_KL','Class_A_NT']

df1.Year = df1.Year.str.replace(' ', '')
df1.Class_A_HK = df1.Class_A_HK.str.replace(' ', '')
df1.Class_A_KL = df1.Class_A_KL.str.replace(' ', '')
df1.Class_A_NT = df1.Class_A_NT.str.replace(' ', '')

df1["Year"] = pd.to_numeric(df1["Year"]).astype(int)
df1["Month"] = pd.to_numeric(df1["Month"]).astype(int)
df1["Class_A_HK"] = pd.to_numeric(df1["Class_A_HK"]).astype(int)
df1["Class_A_KL"] = pd.to_numeric(df1["Class_A_KL"]).astype(int)
df1["Class_A_NT"] = pd.to_numeric(df1["Class_A_NT"]).astype(int)

df1['HKD/sqm (Class A)'] = df1[['Class_A_HK','Class_A_KL','Class_A_NT']].mean(axis=1)
df1=df1.drop(['Class_A_HK','Class_A_KL','Class_A_NT'], axis=1)


#Read Monthly Median Household Income (MMHI)
df2=pd.read_csv("/Users/kwanw4/Downloads/FINA2390_Project 3/MMHHI.csv")
df2=df2[df2.columns[2:5]]
df2=df2.dropna(axis=0, how='any')
df2.columns =['Year', 'Month','MMHHI']
df2["Year"] = pd.to_numeric(df2["Year"]).astype(int)
df2["Month"] = pd.to_numeric(df2["Month"]).astype(int)


#Read Best-lending rate
df3=pd.read_csv("/Users/kwanw4/Downloads/FINA2390_Project 3/best_lending_rate.csv")
df3.drop(df3.head(243).index,inplace=True)
df3.drop(df3.tail(31).index,inplace=True)
df3[df3.columns[0]]=df3[df3.columns[0]].fillna(method="ffill")
df3.drop(df3.columns[2:10], axis=1, inplace=True)
df3=df3.dropna(axis=1, how='any')
df3.columns =['Year', 'Month','r']
df3['Month']=pd.to_datetime(df3.Month, format='%b').dt.month
df3["Year"] = pd.to_numeric(df3["Year"]).astype(int)
df3["Month"] = pd.to_numeric(df3["Month"]).astype(int)
df3["r"] = pd.to_numeric(df3["r"])



# SECTION 2 FINANCIAL ANALYSIS
#-----------------------------------------------------------------------------


data_frames = [df1,df2,df3]
df=reduce(lambda left,right:pd.merge(left,right, on=['Year','Month'], how='outer'),data_frames)

#Household can commit 1/3 of its income to mortgage payment
df['Affordable Mortgage Payment'] = df['MMHHI']/3
#Calculate affordable mortgage loan value by PV model (PMT=mortgage payment, period = 20 years, mortgage rate r= prime rate less 2.5%)
df['Affordable 20-year Mortgage Loan'] = npf.pv((df['r']/100-0.025)/ 12, nper=240, pmt=-df['Affordable Mortgage Payment'])

#Calculate down payment which is 30% of loan value
df['Affordable Home Price with 30% Down payment']=df['Affordable 20-year Mortgage Loan']/0.7

#Calculate actual home price, ie. HKD.sqm * 40sq (Class A)
df['Actual Home Price']=df['HKD/sqm (Class A)']*40

#Calculate size of bubble which is actual home price less affordable home price
df['Bubble']=df['Actual Home Price']-df['Affordable Home Price with 30% Down payment']

#The probability of bubble sustaining
df=df.assign(Probability_burst=(1+((df.r-2.5)/100))*df.Bubble/df.Bubble.shift(-12))
df['Probability_burst'].where(df['Probability_burst'] <= 1, 1, inplace=True)
df['Probability_burst'].where(df['Probability_burst'] >= 0, 0, inplace=True)




# SECTION 3 DATA VISUALIZATION
#-----------------------------------------------------------------------------

#Display Bubble
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df['Actual Home Price'].max()
df['Actual Home Price'].min()
x = np.array(df['Date'])
y1 = np.array(df['Actual Home Price'])
y2 = np.array(df['Affordable Home Price with 30% Down payment'])


plt.plot(x,y1,label="Actual Home Price")
plt.plot(x,y2,label="Affordable Home Price")
plt.title("Affordable Home Price vs. Actual Home Price")
plt.xlabel("Year")
plt.ylabel("Price(HKD)")
plt.legend(loc="upper left")
plt.grid(True)
plt.ylim(top=8000000) 
plt.ylim(bottom=500000) 

#Display probability of bubble burst
plt.figure()
x_bub = np.array(df['Date'])
y_bub = np.array(df['Probability_burst'])
plt.scatter(x_bub,y_bub)

plt.title("Market Expectation Proxy")
plt.xlabel("Year")
plt.ylabel("Probability of bubble sustaining")
plt.grid(True)
plt.show()