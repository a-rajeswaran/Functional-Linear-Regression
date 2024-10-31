# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:33:48 2024

@author: a-raj
"""

import numpy as np
import pandas as pd
import math


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
#df['Jour'] = df['Jour'].astype(int)

df['Year'] = df['Date'].dt.year

theta = 30
sub_kernels_list = []
year_list = [pd.Timestamp(year=i, month=1, day=1).year for i in range(1999, 2023)]
n = len(year_list)

for i in range(1,25):
    X = np.zeros((n ,12))
    Y = np.zeros((n,1))
    for j in range(12):
        df1 = df[df["Jour"]==str(i + j*theta)]
        year_count = 0
        df_count = 0 
        while year_count < n:
            if(df_count < len(df1)):
                if(df1.iloc[df_count,0].year==year_list[year_count]):
                    X[year_count][j] = df1.iloc[df_count,2]
                    df_count +=1
                    year_count +=1
                else:
                    X[year_count][j] = np.nan
                    year_count +=1
            else:
                X[year_count][j] = np.nan
                year_count +=1
        rows_to_delete = []
        for k in range(X.shape[0]):
            bool_nan = any(math.isnan(x) for x in X[k])
            if bool_nan==True:
                rows_to_delete.append(k)
        X_train = np.delete(X,rows_to_delete,axis=0)
        
            
                

            
    
                
                
        