#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def main():
    

 # ----------------------------------NAVBAR -------------------------------------------------------------   
    


 # ----------------------------------PAGES -------------------------------------------------------------     

    local_css("style2.css")

    button_clicked = st.button("OK")
    button_clicked = st.button("KO")
    button_clicked = st.button("ignore")
            
if __name__ == '__main__':
    main()

