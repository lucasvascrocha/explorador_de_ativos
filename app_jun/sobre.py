import streamlit as st
from PIL import Image

def sobre_bix():
    st.title('')  

    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        st.title("Quem somo nós")
    with col3:
        st.write("")             

    image = Image.open('images/sobre_nos.png')
    st.image(image, use_column_width=True)
    st.write('Somos uma consultoria de dados, business intelligence e data science com foco em negócios e na conquista de resultados. Utilizamos as principais ferramentas e tecnologias do mercado no desenvolvimento de projetos que ajudem os nossos clientes a extrair o máximo de valor de seus dados e se transformar digitalmente. Dessa forma, acreditamos que se otimiza a gestão de processos e pessoas, o que facilita e melhora a vida de todos.')
    st.write('A BIX Tecnologia foi fundada na cidade de Florianópolis em 2014 pelo engenheiro Felipe Santos Eberhardt, que percebeu a necessidade do mercado de começar a usar dados para a criação de estratégias mais eficientes e tomada de decisões assertivas. Sem investidores ou aceleradoras, começou a conquistar clientes, aumentar a equipe e desenvolver diversos projetos de inteligência de negócios.')

    st.header('')
    st.subheader('Entre em contato conosco')
    image = Image.open('images/contato.png')
    st.image(image, use_column_width=True)


    
    st.write('Telefone: (48) 99659 5490 / (47) 99981 0094')
    st.write('Email : contato@bixtecnologia.com.br')