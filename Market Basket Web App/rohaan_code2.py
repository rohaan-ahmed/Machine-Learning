#  Apriori

import streamlit as st               
import numpy as np
import pandas as pd
import os

st.title('Generating a Market Basket Recommendation System ## ')

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Select the CSV Raw Data File", type="csv", encoding = "ISO-8859-1")
st.subheader('Uploaded file must have a column with the title \'Items\'')

if uploaded_file:
    data_load_state = st.text('Data Loading Initiated. This may take some time...')
    dataframe = pd.read_csv('historical_data.csv', encoding = "ISO-8859-1")
    dataframe = dataframe['Items']
        
    st.subheader('Raw Data')
    st.text(str(len(dataframe)) + ' Rows')
    st.write(dataframe)
    
    df = dataframe[1:]
    df = df.fillna('NEW_TRANSACTION')
    df = df.values
    df = pd.DataFrame(df)
    df = df[0]
    
    data_load_state.text('Data Loading Completed')
    
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
    
    @st.cache
    def transactions_list(dataset):
        transactions = []
        
        for i in range (0, dataset.shape[0]):
            transactions.append([str(dataset.values[i,j]) for j in range (0, dataset.shape[1])])
        
        for i in range (0, len(transactions)):
            for j in range(0, len(transactions[i])):
                if (transactions[i][j] == str(np.nan)):
                    del transactions[i][j:]
                    break
        return transactions
    
    data_load_state = st.text('Data Transformation Initiated. This may take some time...')
    transactions = transactions_list(dataset)
    data_load_state.text('Data Transformation Completed')
        
    transactions_df = pd.DataFrame(transactions)
    st.subheader('Transaction History')
    st.text(str(len(transactions_df)) + ' Rows')
    st.write(transactions_df)
    
    # Training Apriori on dataset
    
    st.subheader('\nCalculating Items Most Commonly Bought Together\n')
    st.text('Based on the information provided in the Raw Data File, \
            \nwe will now calculate which Items are most frequently bought together.\n')
    
    minimum_support = 0.002
    st.text('Minimum Support: ' + str(minimum_support))
    # minimum_support = st.slider('Minimum Support', min_value = 0.0, max_value = 0.1, value = 0.002, step=0.001)  # min: 0, max: 1, default: 0.002
    minimum_confidence = 0.3
    st.text('Minimum Confidence: ' + str(minimum_confidence))
    # minimum_confidence = st.slider('Minimum Confidence', min_value = 0.0, max_value = 1.0, value = 0.3, step=0.1)
    # minimum_lift = 3
    minimum_lift = st.slider('Select: Minimum Lift:', min_value = 1.0, max_value = 5.0, value = 3.0, step=1.0)
    # minimum_length = 2
    minimum_length = st.slider('Select: Minimum Length:', min_value = 1.0, max_value = 5.0, value = 2.0, step=1.0)
    
    from apyori import apriori
    rules = apriori(transactions, min_support = minimum_support, min_confidence = minimum_confidence, min_lift = minimum_lift, min_length = minimum_length)
    
    # Visualizing the results
    results = list(rules)
    
    if len(results) != 0:
        # Creating a dataframe object from list
        df_results = pd.DataFrame(results)
        df_items = pd.DataFrame(list(df_results['items']))
        df_items['support'] = pd.DataFrame(list(df_results['support']))
        
        st.subheader('Items Most Commonly Bought Together')
        st.subheader('Results: ' + str(len(df_items)) + ' combinations')
        st.write(df_items)
        
        df_items.to_csv('Output_Rules.csv')
    else:
        st.subheader('No Results to Display')