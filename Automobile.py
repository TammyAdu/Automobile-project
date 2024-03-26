import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import streamlit as st
import joblib

data = pd.read_csv('Automobile_project.csv')
model = joblib.load('automobile_model.pkl')


st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: helvetica'>PRICE PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin:-#0071AD; color;text-align: center; font-family: cursive '>Built By ELIZABETH TANIMOLA ADU<h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (4).png')

st.markdown("<h4 style = 'margin: -#0071AD; color:blue; text-align: center; font-family: helvetica '>Project Overview</h4>", unsafe_allow_html = True)

st.write("This project aims to address these challenges by leveraging machine learning techniques to develop a predictive model capable of accurately estimating automobile prices based on relevant features. By utilizing historical data and advanced algorithms, the model can analyze various factors influencing pricing, such as vehicle specifications, market trends, and demand-supply dynamics. Ultimately, this approach enables stakeholders to make more informed pricing decisions, optimizing sales strategies and maximizing profitability.")

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width= True)
st.sidebar.image('pngwing.com (5).png',caption = 'Welcome Dear User')


curb_weight = st.sidebar.number_input('curb_weight', data['curb_weight'].min(), data['curb_weight'].max())
symboling = st.sidebar.number_input('symboling', data['symboling'].min(), data['symboling'].max())
normalized_losses = st.sidebar.number_input('normalized_losses', data['normalized_losses'].min(), data['normalized_losses'].max())
height = st.sidebar.number_input('height', data['height'].min(), data['height'].max())
make = st.sidebar.selectbox('make', data['make'].unique())
body_style = st.sidebar.selectbox('body_style', data['body_style'].unique())

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'margin: -30px; color: green and white; text-align: center; font-family: helvetica '>Input Variables</h4>", unsafe_allow_html = True)

inputs = pd.DataFrame()
# inputs['curb_weight'] = [curb_weight]
inputs['symboling'] = [symboling ]
inputs['normalized_losses'] = [normalized_losses]
inputs['height'] = [height]
inputs['make'] = [make]
inputs['body_style'] = [body_style]

make_trans = joblib.load('make_encoder.pkl')
body_style_trans = joblib.load('body_style_encoder.pkl')

inputs['make'] = make_trans.transform(inputs[['make']])
inputs['body_style'] = body_style_trans.transform(inputs[['body_style']])

prediction_button = st.button('Predict Price')
if prediction_button:
   predicted = model.predict(inputs)
   st.success(f'The Price predicted for your business is {predicted[0].round(2)}')
