#  Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
               
dataframe = pd.read_csv('historical_data.csv', encoding = "ISO-8859-1")
#dataframe = pd.read_excel('historical_data3_excel.xls')
#dataframe = pd.read_excel('historical_data_onecolumn.csv')
dataframe = dataframe['Items']
df = dataframe[1:]
df = df.fillna('NEW_TRANSACTION')
df = df.values
df = pd.DataFrame(df)
df = df[0]

for i in range (0, len(df)):
    if (df[i] == 'NEW_TRANSACTION'):
        text = ''
        j = 1
        while (df[i+j] != 'NEW_TRANSACTION' and i+j < len(df)-1):
            text = (text + df[i+j] + ',')
            j = j + 1
        if (text != ''):
            df[i] = text
    else:
        df[i] = np.nan
df = df.dropna()
dataset = df.str.split(',', expand=True)
dataset = dataset.replace(to_replace = '', value = np.nan)
dataset.fillna(value = np.nan, inplace=True)

transactions = []

for i in range (0, dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range (0, dataset.shape[1])])

for i in range (0, len(transactions)):
    for j in range(0, len(transactions[i])):
        if (transactions[i][j] == str(np.nan)):
            del transactions[i][j:]
            break
            
# Training Apriori on dataset

minimum_support = 0.002

from apyori import apriori
rules = apriori(transactions, min_support = minimum_support, min_confidence = 0.3, min_lift = 3, min_length = 2)

# Visualizing the results
results = list(rules)

# Creating a dataframe object from list
df_results = pd.DataFrame(results)
df_items = pd.DataFrame(list(df_results['items']))
df_items['support'] = pd.DataFrame(list(df_results['support']))

df_items.to_csv('Output_Rules.csv')