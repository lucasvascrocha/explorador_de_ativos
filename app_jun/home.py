import streamlit as st
from PIL import Image

def home():
    st.title('')               
    st.title('Leitura de caracteres')
    st.subheader('Faça leitura de caracteres através de imagens ou vídeos!')
    image = Image.open('images/home.png')
    st.image(image, use_column_width=True)

    st.subheader('Para quais aplicações posso utilizar o leitor de caracteres?')
    st.write('Para qualquer situação onde se queira fazer leitura de caracteres através de imagens ou de vídeos!')
    st.write('Exemplos de uso são empresas com linha de produção que precisam fazer leitura de códigos de produto, leitura de containers, peças, etc....')

    st.title('')
    st.subheader('Como funciona?')
    st.write('O algoritimo analisa o vídeo ou imagem e pode entregar imagens com a leitura dos caracteres e um arquivo contendo os textos encontrados.')
    st.write('Os arquivos com a leitura dos caracteres podem ser integrados com ferramentas de terceiros ou adicionadas a um banco de dados por exemplo.')

    st.subheader('Experimente, é gratuito!')
    st.write('Vá até a aba "Leitura de caracteres" e faça um teste com nossos dados ou entre com um vídeo próprio e teste a aplicação agora mesmo!')

        