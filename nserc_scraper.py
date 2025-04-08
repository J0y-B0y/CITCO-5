import requests
import pandas as pd
import json
import time
from datetime import datetime

fields = ['Name-Nom', 'AreaOfApplicationCode']

df = pd.read_csv('2023.csv', encoding='latin-1', usecols=fields)

filterBySubject = df[(df['AreaOfApplicationCode'] == 800)] # 800 is the information systems code

df_Out = filterBySubject[['Name-Nom']]

print (df_Out) #only returns names

df_Out.to_csv('names.csv')