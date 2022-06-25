#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
st.set_page_config(  # Alternate names: setup_page, page, layout
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)

import matplotlib
matplotlib.use('Agg')

from streamlit_option_menu import option_menu

import warnings
warnings.filterwarnings('ignore')
 
import datetime as dt 
dia = dt.datetime.today().strftime(format='20%y-%m-%d')

import style as style
import pag1 as pag1
import pag2 as pag2
import pag3 as pag3
import pag4 as pag4
import pag5 as pag5
import pag1_02 as pag1_02

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)




#style.set_background('images/bkgold.png')
#style.set_css()

    #     st.markdown(
    # """
    # <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    # <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    # """,unsafe_allow_html=True
    #     )


def main():
    

 # ----------------------------------NAVBAR -------------------------------------------------------------   
    # n_sprites = option_menu(None,["Home", "Login", "Cadastro"],
    #                         icons=['', '', ''],
    #                         default_index= 0, menu_icon="app-indicator", orientation='horizontal',   #
    #                         styles={
    #         "container": {"padding": "10!important", "background-color": "#f0eeee" }, # ,"background-size": "cover","margin": "0px"},
    #         #"container": {"padding": "0", "background-color": "#fafafa","background-size": "cover","margin": "0px"},
    #         #"icon": {"color": "orange", "font-size": "25px"}, 
    #         #"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    #         "nav-link": {"font-size": "16px", "text-align": "center", "--hover-color": "#eee","font-weight": "bold"}, #,"position": "relative","display": "inline"},
    #         "nav-link-selected": {"background-color": "#4E90FF"},
    #     }
    #     ) 
    
    with st.sidebar:
        n_sprites = option_menu('Menu',["Home","Login","Cadastro","Análise técnica e fundamentalista", "Comparação de ativos", "Descobrir novos ativos", "Rastreador de trade", "Análise de carteira e previsão de lucro"],
                            icons=['bar-chart', 'book', 'bullseye', 'binoculars','cash-coin'],
                            default_index=0, menu_icon="app-indicator",   #orientation='horizontal',
                            styles={
            "container": {"padding": "2!important", "background-color": "#ffffff" }, # ,"background-size": "cover","margin": "0px"},
            #"container": {"padding": "0", "background-color": "#fafafa","background-size": "cover","margin": "0px"},
            #"icon": {"color": "orange", "font-size": "25px"}, 
            #"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link": {"font-size": "12px", "text-align": "left", "--hover-color": "#eee","font-weight": "bold"}, #,"position": "relative","display": "inline"},
            "nav-link-selected": {"background-color": "#4E90FF"},
        }
        ) 

 # ----------------------------------PAGES -------------------------------------------------------------     

    if n_sprites == "Análise técnica e fundamentalista":
        local_css("style_1.css")      
        pag1_02.analise_tecnica_fundamentalista2()

    if n_sprites == "Comparação de ativos":
        local_css("style_2.css")   
        pag2.comparacao_ativos()

    if n_sprites == "Descobrir novos ativos":
        pag3.descobrir_ativos()

    if n_sprites == "Rastreador de trade":
        pag4.rastreador()     

    if n_sprites == "Análise de carteira e previsão de lucro":
        pag5.analise_carteira()
        
if __name__ == '__main__':
    main()

