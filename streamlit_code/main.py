# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 13:41:21 2021

@author: 1353271
"""

import pandas as pd
from prediction_module import predict_topics, predict_topics_list, predict_single
import streamlit as st
labels_ = ['Computer Science', 'Physics', 'Mathematics','Statistics', 'Quantitative Biology', 'Quantitative Finance']

st.set_page_config(page_title="whitepaper_classifier",
                   page_icon="📝",
                   layout="wide")

# predict_topics(raw_testing_data)
# predict_topics_list(raw_testing_data)
# row = 100
# predict_single(raw_testing_data.loc[row,'TITLE'],raw_testing_data.loc[row,'ABSTRACT'])

st.title('Whitepaper classifier')
st.text('https://www.kaggle.com/vin1234/janatahack-independence-day-2020-ml-hackathon?select=train.csv')

st.header('Single paper classification')
st.subheader('Enter title and abstract to get topics')
# st.button('Get Topics')

with st.form("form_single_paper"):
    title = st.text_input('Title', value='')
    abstract = st.text_input('Abstract', value='')
    # Every form must have a submit button.
    submitted = st.form_submit_button("Get Topics")
    if submitted:
        topics = predict_single(title, abstract)
        st.write('This whitepaper belongs to ',topics)


st.header('Bulk paper classification')
st.subheader('upload csv files to get predictions')
# st.button('Get Topics')

@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

@st.cache
def get_predictions(raw_data_file, topics=False):
    if topics:
        predictions = predict_topics_list(raw_testing_file)
    else:
        predictions = predict_topics(raw_testing_file)
    return predictions



with st.form("form_csv_file"):
    input_file = st.file_uploader("upload comma separated file having title and abstract columns. Column names: TITLE, ABSTRACT", accept_multiple_files=False)
    get_topics_list = st.radio('Do you want to get topics list in the prediction file?', ('yes', 'no'))
    # Every form must have a submit button.
    submitted = st.form_submit_button("Get Prediction file")
    if submitted:
        try:
            raw_testing_file = pd.read_csv(input_file)
            output_df = raw_testing_file.copy()
            if get_topics_list=='yes':
                predictions = get_predictions(raw_testing_file, topics=True)
                labels_list = labels_ + ['topics']
            else:
                predictions = get_predictions(raw_testing_file, topics=False)
                labels_list = labels_
                
            output_df.loc[:,labels_list] = predictions
            st.dataframe(output_df)
            success = True
        except:
            st.warning("Something is wrong with the file. Try again...")
            success = False
        
if submitted and success:
    csv = convert_df(output_df)
    st.download_button(label="Press to Download",
                       data=csv,
                       file_name='predictions_file.csv',
                       mime='text/csv')