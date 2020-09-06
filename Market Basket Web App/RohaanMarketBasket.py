#  Apriori

import streamlit as st               
import numpy as np
import pandas as pd
import base64
import json
import pickle
import uuid
import re

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

st.title('Generating a Market Basket Recommendation System ## ')

st.subheader('This page uses the \"Association Rule Algorithm\" to find the Items Most Commonly Bought Together\n')
st.text('This is calculated based on the information provided in the Raw Historical Data file')
    
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Select the CSV Raw Historical Data File", type="csv", encoding = "ISO-8859-1")
st.text('Requirements for the Raw Historical Data file are listed at the bottom of the page')

if uploaded_file:
    # data_load_state = st.text('Data Loading Initiated. This may take some time...')
    dataframe = pd.read_csv(uploaded_file, encoding = "ISO-8859-1")
    dataframe = dataframe['Items']
        
    st.subheader('Raw Historical Data')
    st.text(str(len(dataframe)) + ' Rows')
    st.write(dataframe)
    
    df = dataframe[1:]
    df = df.fillna('NEW_TRANSACTION')
    df = df.values
    df = pd.DataFrame(df)
    df = df[0]
    
    # data_load_state.text('Data Loading Completed')
    
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
    st.subheader('--------------------------------------------------------------')
    st.subheader('Consolidated Transaction History')
    st.text(str(len(transactions_df)) + ' Rows')
    st.write(transactions_df)
    
    # Training Apriori on dataset
    
    st.subheader('--------------------------------------------------------------')
    # minimum_support = 0.002
    st.subheader('Select the desired "Minimum Support:\"')
    st.text('This is a rough measure of how \"confident\" you want the prediction to be')
    minimum_support = st.slider('Minimum Support', min_value = 0.0, max_value = 10.0, value = 0.2, step=0.1)  # min: 0, max: 1, default: 0.002
    minimum_support = minimum_support/100
    
    minimum_confidence = 0.3
    #st.text('Minimum Confidence: ' + str(minimum_confidence))
    # minimum_confidence = st.slider('Minimum Confidence', min_value = 0.0, max_value = 1.0, value = 0.3, step=0.1)
    minimum_lift = 3
    # minimum_lift = st.slider('Select: Minimum Lift:', min_value = 1.0, max_value = 5.0, value = 3.0, step=1.0)
    minimum_length = 2
    # minimum_length = st.slider('Select: Minimum Length:', min_value = 1.0, max_value = 5.0, value = 2.0, step=1.0)
    
    from apyori import apriori
    rules = apriori(transactions, min_support = minimum_support, min_confidence = minimum_confidence, min_lift = minimum_lift, min_length = minimum_length)
    
    # Visualizing the results
    results = list(rules)
    
    if len(results) != 0:
        # Creating a dataframe object from list
        df_results = pd.DataFrame(results)
        df_items = pd.DataFrame(list(df_results['items']))
        df_items['confidence'] = pd.DataFrame(list(df_results['support']))
        df_items['confidence'] = df_items['confidence']*100
        
        df_items = df_items.sort_values(by=['confidence'], ascending = False)
        df_items = df_items.reset_index(drop=True)
        
        st.subheader('List of Items Most Commonly Bought Together')
        st.subheader('Results: ' + str(len(df_items)) + ' combinations')
        st.write(df_items)
        
        if st.button('Download as CSV'):
            tmp_download_link = download_button(df_items, 'MarketBasketRules.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
    
        # df_items.to_csv('Output_Rules.csv')
    else:
        st.subheader('No Results to Display')

st.subheader('How to use this Page\n')

st.subheader('The intent of this simple app is to allow retailers to generate insights into which items are most \
             frequently bought together. The results of this app can be used to generate \'upselling\' insights.\n' + \
             'For example, if the final result shows that "milk" and "cookies" are usually bought together, a \
                     retailer may use this information to place these items next to each other, or offer these items as "addons" at checkout\n\n')

st.subheader('--------------------------------------------------------------')

st.text('\n\n' + 'The Raw historical Data file must conain a column with the title \'Items\'\n\
and encoding ISO-8859-1')
    
st.subheader('Created for fun using Streamlit, hosted on AWS, by Rohaan Ahmed, in 2020')
st.text('Code: https://github.com/rohaan-ahmed/Master-Repository/')