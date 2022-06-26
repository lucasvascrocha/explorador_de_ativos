 # ----------------------------------LIBS -------------------------------------------------------------   
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
import home as home
import login as login
import pag1 as pag1
import pag2 as pag2
import pag3 as pag3
import pag4 as pag4
import pag5 as pag5


 # ----------------------------------DEFS -------------------------------------------------------------   

#carrega os arquivos css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():

    #css comum para todas páginas (menu)
    #local_css("style_0.css")  

    #esconder botão de menu e marca dágua no rodapé
    style.hidden_menu_and_footer()

    #cabeçalho detalhe superior da página 
    style.headerstyle()

 # ----------------------------------MENU -------------------------------------------------------------   

    with st.sidebar:
        style.sidebarwidth()    
        n_sprites = option_menu('Menu',["Home","Login","Análise técnica", "Comparar ações", "Análise fundamentalista", "Rastrear ações", "Previsão de lucro"],
                            icons=['house','person','bar-chart', 'book', 'bullseye', 'binoculars','cash-coin'],
                            default_index=0, menu_icon="app-indicator",   #orientation='horizontal',
                            styles={
            "container": {"padding": "2!important", "background-color": "#ffffff" }, # ,"background-size": "cover","margin": "0px"},
            "nav-link": {"font-size": "12px", "text-align": "left", "--hover-color": "#4a7198","font-weight": "bold"}, #,"position": "relative","display": "inline"},
            "nav-link-selected": {"background-color": "#4a7198"},
             }) 

 # ----------------------------------PAGES -------------------------------------------------------------     
    if n_sprites == "Home":
        #local_css("style_1.css")      
        home.initial_page()
    
    if n_sprites == "Login":
        #local_css("style_login.css")      
        login.login_section()

    if n_sprites == "Análise técnica":
        #local_css("style_1.css")      
        pag1.analise_tecnica_fundamentalista()

    if n_sprites == "Comparar ações":
        #local_css("style_2.css")   
        pag2.comparacao_ativos()

    if n_sprites == "Análise fundamentalista":
        #local_css("style_3.css")   
        pag3.descobrir_ativos()

    if n_sprites == "Rastrear ações":
        #local_css("style_4.css")   
        pag4.rastreador()     

    if n_sprites == "Previsão de lucro":
        #local_css("style_5.css")  
        pag5.analise_carteira()
        
if __name__ == '__main__':
    main()

