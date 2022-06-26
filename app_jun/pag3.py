import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
from collections import OrderedDict

import scrap as scraping

def flatten(d):
    '''
    Flatten an OrderedDict object
    '''
    result = OrderedDict()
    for k, v in d.items():
        if isinstance(v, dict):
            result.update(flatten(v))
        else:
            result[k] = v
    return result

def descobrir_ativos():
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            #st.image('https://media.giphy.com/media/3ohs4gux2zjc7f361O/giphy.gif', width=400)
            st.write("")
        with col3:
            st.write("")

        #st.image('https://media.giphy.com/media/3ohs4gux2zjc7f361O/giphy.gif', width=400)    
        #image = Image.open('imagens/logo.jpg')
        #st.image(image, use_column_width=True)                       
        st.title('Descobrir novos ativos')

        PL_mínimo = int(st.number_input(label='PL_mínimo',value=10))
        PL_máximo = int(st.number_input(label='PL_máximo',value=15))
        PVP_mínimo = int(st.number_input(label='PVP_mínimo',value=0.7))
        PVP_máximo = int(st.number_input(label='PVP_máximo',value=1.5))
        DY_mínimo = int(st.number_input(label='DY_mínimo',value=4))
        DY_máximo = int(st.number_input(label='DY_máximo',value=30))

        lista = scraping.get_data()
        todos = pd.DataFrame(flatten(lista).keys()).transpose()
        todos.columns = todos.iloc[0]

        for i in range(len(lista)):
          todos = pd.concat([todos,pd.DataFrame(lista[i]).transpose()])

        todos = todos.iloc[1:]
        todos['P/L'] = todos['P/L'].str.replace('.','')
        todos['DY'] = todos['DY'].str.replace('%','')
        todos['Liq.2m.'] = todos['Liq.2m.'].str.replace('.','')
        todos['Pat.Liq'] = todos['Pat.Liq'].str.replace('.','')
        todos = todos.replace(',','.', regex=True)
        todos = todos.apply(pd.to_numeric,errors='ignore')


        if st.checkbox("Filtrar"):

            st.dataframe(todos.loc[(todos['P/L']>= PL_mínimo) & (todos['P/L']<= PL_máximo) & (todos['P/VP']>= PVP_mínimo) & (todos['P/VP']<= PVP_máximo) & (todos['DY']>= DY_mínimo) & (todos['DY']<= DY_máximo)])
            
            