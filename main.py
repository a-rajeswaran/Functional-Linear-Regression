# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:32:57 2024

@author: a-raj
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def compute_linear_estimator(X,Y):
    X_T=np.transpose(X)
    B=np.dot(X_T,X)
    B=np.linalg.inv(B)
    B=np.dot(B,X_T)
    B=np.dot(B,Y)
    return B

df_glacier = pd.read_csv('Glacier Blanc.txt', delim_whitespace=True)
df_glacier['année'] = pd.to_datetime(df_glacier['année'].astype(str) + "/09/26", format='%Y/%m/%d')
df_glacier[['accumulation', 'ablation', 'bilan']] = df_glacier[['accumulation', 'ablation', 'bilan']].astype(float)

df = pd.read_csv('Température Embrun.txt', delim_whitespace=True)
df["Date"] = pd.to_datetime(df['Date'].astype(str), format='%Y/%m/%d')
df['Valeur'] = df['Valeur'].str.replace(',', '.')
df['Valeur'] = df['Valeur'].astype(float)
df['Year'] = df['Date'].dt.year

theta = 30
sub_kernels_list = []
year_list = [pd.Timestamp(year=i, month=1, day=1).year for i in range(1999, 2023)]
n = len(year_list)

start_date = datetime.strptime("1999/09/17", "%Y/%m/%d")
start = 250
Y = np.array(df_glacier["accumulation"])
n=len(Y)

df = df.iloc[250:]
df = df[df['Jour'] != 'Leap']

end_date = pd.to_datetime("2017-09-17")
#df = df[df['Date'] < end_date]



Y = np.array(df_glacier["accumulation"])
for i in range(theta):
    print(i)
    rows_to_delete = []
    X = np.zeros((n ,12))
    for j in range(12):
        if((260 + i + j*theta) % 365 == 0):
            df1 = df[df['Jour'] == '365']
        else:
            df1 = df[df["Jour"]==str((260 + i + j*theta) % 365)]
        boolean = (260 + j*theta) // 365 > 1
        year_count = 0
        df_count = 0 
        while year_count < n:
            if(df_count < len(df1)):
                if(df1.iloc[df_count,0].year==year_list[year_count]):
                    X[year_count][j] = df1.iloc[df_count + boolean,2]
                    df_count +=1
                    year_count +=1
                    
                else:
                    X[year_count][j] = np.nan
                    year_count +=1
            else:
                X[year_count][j] = np.nan
                year_count +=1  
    for k in range(X.shape[0]):
        bool_nan = any(math.isnan(x) for x in X[k])
        if bool_nan==True:
            rows_to_delete.append(k)
    X_train = np.delete(X,rows_to_delete,axis=0)
    Y_train = np.delete(Y, rows_to_delete, axis=0)
    print(X_train.shape)
    subregression = compute_linear_estimator(X_train, Y_train)
    sub_kernels_list.append(subregression)
    
B=[]
for k in range(12):
    for l in range(len(sub_kernels_list)):
        B.append(sub_kernels_list[l][k])
        
date1 = pd.to_datetime("1999-09-16")
date2 = pd.to_datetime("2000-09-16")  
liste_result =[]
X_test = df[(df['Date'] > date1) & (df['Date'] <= date2)]
result = 0
for k in range(260,620):
    if k==365:
        temps = df.loc[df["Jour"] == str(365), "Valeur"]
    else:
        temps = df.loc[df["Jour"] == str(k%365), "Valeur"]
    if not temps.empty:
        temps = temps.iloc[0]
        result = result +temps*B[k-260]
    else:
        print(k)
result=result/30
liste_result.append(result)





"""
for i in range(23):
    date1 = date1 + pd.DateOffset(years=i)
    date2 = date2 + pd.DateOffset(years=i)
    X_test = df[(df['Date'] > date1) & (df['Date'] <= date2)]
    
    result = 0
    for k in range(260,620):
        temps = df.loc[df["Jour"] == str(k), "Valeur"]
        if not temps.empty:
            temps = temps.iloc[0]
            result = result +temps*B[k-260]
        else:
            right_bool = False
            left_bool = False
            r = 1
            l = 1
            while right_bool==False or left_bool==False:
                t1 = df.loc[df["Jour"] == str(k+r), "Valeur"]
                t2 = df.loc[df["Jour"] == str(k-l), "Valeur"]
                if not t1.empty:
                    right_bool=True
                else:
                    r=r+1
                    print(r)
                if not t2.empty:
                    left_bool=True
                else:
                    l=l+1 
                    print(l)
            result = result + B[k - 260]*(t1.iloc[0]+t2.iloc[0])/2
    liste_result.append(result)
"""   

    
    



    

    








































"""
for j in range(12):
    df1 = df[df["Jour"]==str((260 + j*theta) % 365)]
    boolean = (260 + j*theta) // 365 >= 1
    year_count = 0
    df_count = 0 
    while year_count < n:
        if(df_count < len(df1)):
            if(df1.iloc[df_count,0].year==year_list[year_count]):
                X[year_count][j] = df1.iloc[df_count + boolean,2]
                df_count +=1
                year_count +=1
            else:
                X[year_count][j] = np.nan
                year_count +=1
        else:
            X[year_count][j] = np.nan
            year_count +=1

"""







"""
for t in range(theta):
    X = np.zeros((len(Y) ,12))
    for i in range(len(Y)):
        for j in range(12):
            deltaT = relativedelta(years=i, days= t + j*theta)
            if(df.iloc[start + t + j*theta, 0] + relativedelta(years=i) == start_date + deltaT ):
                X[i][j] = df[df['Date'] == start_date + deltaT]['Valeur'].values[0]
            else:
                X[i][j] = np.nan
        rows_to_delete = []
        for k in range(X.shape[0]):
            bool_nan = any(math.isnan(x) for x in X[k])
            if bool_nan==True:
                rows_to_delete.append(k)
        X_train = np.delete(X,rows_to_delete,axis=0)
        Y_train = np.delete(Y, rows_to_delete, axis=0)
"""
        
        
    
    
    
        

        
        
        
        
            
                
            
            
        
        
    










