# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:07:45 2024

@author: a-raj
"""

import numpy as np
import pandas as pd
import math
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
#df['Jour'] = df['Jour'].astype(int)

df['Year'] = df['Date'].dt.year

theta = 30
sub_kernels_list = []
year_list = [pd.Timestamp(year=i, month=1, day=1).year for i in range(1999, 2024)]
n = len(year_list)

date_initiale = datetime.strptime("09/10/2023", "%d/%m/%Y")

# Durée à ajouter (30 jours)
duree = timedelta(days=200)

# Addition de la durée à la date initiale
nouvelle_date = date_initiale + duree

# Affichage de la nouvelle date
print(nouvelle_date.strftime("%d/%m/%Y"))

Y = np.array(df_glacier["accumulation"])



start_date = datetime.strptime("1999/01/01", "%Y/%m/%d")
x=df[df['Date'] == df.iloc[0 + 2*theta, 0] + relativedelta(years=3)]['Valeur'].values[0]
print(df[df['Date'] == df.iloc[0 + 2*theta, 0] + relativedelta(years=3)]['Valeur'].values[0])


print(df.iloc[250,2])
df2 = df[df["Jour"]==str((260 + 11*theta) % 365)]
