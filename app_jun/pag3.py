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
        #código para ativar bootstrap css
        st.markdown(
        """
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        """,unsafe_allow_html=True
        )  

        col1, col2,col3 = st.columns([0.1,0.4,0.1])   
        with col2:                        
            st.title('Descobrir novos ativos com filtros fundamentalistas')

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
        todos['DY'] = todos['DY'].str.replace('.','').astype(float)/100
        todos.rename(columns={'cotacao': 'Cotação'}, inplace=True)


        if st.button("Filtrar"):

            # st.dataframe(todos.loc[(todos['P/L']>= PL_mínimo) & (todos['P/L']<= PL_máximo) & (todos['P/VP']>= PVP_mínimo) & (todos['P/VP']<= PVP_máximo) & (todos['DY']>= DY_mínimo) & (todos['DY']<= DY_máximo)]
            # .style.format({"Cotação": "{:.2f}", "P/L": "{:.2f}", "P/VP": "{:.2f}", "P/Ativo": "{:.2f}"
            #     , "P/EBIT": "{:.2f}", "P/Ativ.Circ.Liq.": "{:.2f}", "EBITDA": "{:.2f}", "Liq.Corr.": "{:.2f}", "Liq.2m.": "{:.2f}"
            #     , "Pat.Liq": "{:.2f}", "Div.Brut/Pat.": "{:.2f}"                
            #           }))


            st.dataframe(todos.loc[(todos['P/L']>= PL_mínimo) & (todos['P/L']<= PL_máximo) & (todos['P/VP']>= PVP_mínimo) & (todos['P/VP']<= PVP_máximo) & (todos['DY']>= DY_mínimo) & (todos['DY']<= DY_máximo)])

            