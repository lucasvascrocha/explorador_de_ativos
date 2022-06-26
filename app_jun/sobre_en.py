import streamlit as st
from PIL import Image

def sobre_bix():
    st.title('')  

    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        st.title("About us")
    with col3:
        st.write("")             

    image = Image.open('images/sobre_nos.png')
    st.image(image, use_column_width=True)
    st.write('We are a consulting firm focused on data and implementing data science, data engineering, business intelligence, and software development projects to improve business results.')
    st.write("We use the main tools and technologies on the market to develop projects that help our customers extract the maximum value from their data and digitally transform themselves. In this way, we believe that the management of processes and people is optimized, facilitating and improving everyone's lives.")
    st.header('')
    st.subheader('Contact us')
    image = Image.open('images/contato.png')
    st.image(image, use_column_width=True)


    
    st.write('Phone contact: +55 (48) 99659 5490 / +55 (47) 99981 0094')
    st.write('Email : contato@bixtecnologia.com.br')