import streamlit as st
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


import style as style
import home_en as home_en
import ocrreader_en as ocr_en
import sobre_en as sobre_en

import home as home
import ocrreader as ocr
import sobre as sobre


# ----------------------------------SIDEBAR -------------------------------------------------------------
def main():

    #style.set_background('images/bg03.jpg')

    n_sprites_2 = st.sidebar.selectbox(
        "Select language", ("Englsih","Portuguese"))

    if n_sprites_2 == "Englsih":


        st.sidebar.header("OCR - Character reading from images")
        n_sprites = st.sidebar.radio(
            "Choose an option", options=["Home","Character reading","About Bix-tecnologia"], index=0
        )

        #style.spaces_sidebar(15)
        st.sidebar.write('https://www.bixtecnologia.com/')
        image = Image.open('images/logo_sidebar_sem_fundo.png')
        st.sidebar.image(image, use_column_width=True)

        #st.image(image, use_column_width=True)  
        
    # ------------------------------ INÍCIO ANÁLISE TÉCNICA E FUNDAMENTALISTA ----------------------------             

        if n_sprites == "Home":

            home_en.home()

        if n_sprites == "Character reading":

            ocr_en.ocr_reader()

        if n_sprites == "About Bix-tecnologia":

            sobre_en.sobre_bix()        

    
    else:
        st.sidebar.header("Leitor de caracteres")
        n_sprites = st.sidebar.radio(
            "Escolha uma opção", options=["Home","Leitura de caracteres","Sobre a Bix-tecnologia"], index=0
        )

        #style.spaces_sidebar(15)
        st.sidebar.write('https://www.bixtecnologia.com/')
        image = Image.open('images/logo_sidebar_sem_fundo.png')
        st.sidebar.image(image, use_column_width=True)

        #st.image(image, use_column_width=True)  
        
    # ------------------------------ INÍCIO ANÁLISE TÉCNICA E FUNDAMENTALISTA ----------------------------             

        if n_sprites == "Home":

            home.home()

        if n_sprites == "Leitura de caracteres":

            ocr.ocr_reader()

        if n_sprites == "Sobre a Bix-tecnologia":

            sobre.sobre_bix()        


 
        
if __name__ == '__main__':
    main()