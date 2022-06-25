import matplotlib
matplotlib.use('Agg')

import streamlit as st

import pandas as pd

def login_section():

        #código para ativar bootstrap css
    st.markdown(
"""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""",unsafe_allow_html=True
    )  

    users = pd.read_csv('users.csv', sep=';',dtype=str)

    col1, col2, col3,col4,col5 = st.columns([1,1,1,1,1])   
    with col2:  

        st.header('Login')

        username = st.text_input("User Name")
        password = st.text_input("Password",type='password')   

        if 'username' not in st.session_state:
            st.session_state['username'] = [username]

        if 'loged' not in st.session_state:
            st.session_state['loged'] = ''  

        if st.button('Login'):
            if len(users[users['user'].str.contains(username, na=False)]) > 0:
                if len(users[users['pass'].str.contains(password, na=False)]) > 0:
                    st.session_state['loged'] = 'Logado'

            else:
                st.text('Senha incorreta')

            if password == users['pass'][0]:
                st.write('acesso manager')
                st.dataframe(users)


        #esse código iniciará em outras páginas as funções para quem estiver logado
        if st.session_state['loged']:
            st.write(st.session_state['loged'])

    with col4: 

        st.header('Cadastro')

        cad_nome = st.text_input("Nome")
        cad_email = st.text_input("Email")
        cad_username = st.text_input("Cadastre um nome de usuário")
        cad_password = st.text_input("Cadastre uma senha",type='password')
        cad_password_2 = st.text_input("Repita a senha",type='password')  
        cad_premium = 'não'

        df_cad = pd.DataFrame( [[cad_nome,cad_username,cad_password,cad_email, cad_premium]] ,columns=['name','user','pass','email','premium'] )    

        if st.button('Cadastrar'):
            if len(users[users['user'].str.contains(cad_username, na=False)]) == 0:
                dfs = [users,df_cad]
                users = pd.concat( dfs,axis=0,ignore_index=True)
                users.to_csv('users.csv', index=False, sep=';')
                st.text('Cadastro efetuado com sucesso, realize o Login')  
            else:
                st.text('Usuário já existente')         










