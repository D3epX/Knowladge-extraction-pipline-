#this is some streamlit basic examples done while exploring it's docs
import streamlit as st
import numpy as np 
import pandas as pd


st.write("Hello, I'm your AI assistant! How can I help you today?")
df= pd.DataFrame({'x': np.random.ranf(10),
                  'y': np.random.rand(10)})
st.write(df)
st.table(df)
st.line_chart(df)
st.dataframe(df.style.highlight_max(axis=0))
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
st.map(map_data)
z= st.slider('Select a value')
st.write(z, 'squared is', z *z)
