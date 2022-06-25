import matplotlib
matplotlib.use('Agg')

import streamlit as st

import html_home as html_home
from PIL import Image

def initial_page():

        #código para ativar bootstrap css
    st.markdown(
"""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""",unsafe_allow_html=True
    )  

    image = Image.open('images/home_image_2.jpeg')
    st.image(image, use_column_width=True)  