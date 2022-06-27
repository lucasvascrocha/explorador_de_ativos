import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict

import scrap as scraping

#from st_aggrid import AgGrid



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
            st.title('Analisar ativos com filtros e recomendações fundamentalistas')
        st.title("")

        st.subheader('Médias dos principais setores, subsetores e segmentos')

        codigo_nome = pd.read_excel('data/classification_b3.xlsx')

        lista = scraping.get_data()
        todos = pd.DataFrame(flatten(lista).keys()).transpose() 
        todos.columns = todos.iloc[0]

        for i in range(len(lista)):
          todos = pd.concat([todos,pd.DataFrame(lista[i]).transpose()])

        todos = todos.iloc[1:]
        todos = todos.reset_index()

        

        #tratamentos
        todos['P/L'] = todos['P/L'].str.replace('.','')
        todos['DY'] = todos['DY'].apply(lambda x: x[:-1])
        todos['Cresc.5a %'] = todos['Cresc.5a'].apply(lambda x: x[:-1]).str.replace('.','').str.replace(',','.').astype(float)
        todos['ROE'] = todos['ROE'].apply(lambda x: x[:-1]).str.replace('.','').str.replace(',','.').astype(float)
        todos['Pat.Liq'] = todos['Pat.Liq'].str.replace('.','')
        todos = todos.replace(',','.', regex=True)
        todos = todos.apply(pd.to_numeric,errors='ignore')
        todos.rename(columns={'cotacao': 'Cotação'}, inplace=True)
        todos.rename(columns={'index': 'Código'}, inplace=True)

        

        #criação colunas
        todos['VPA'] = todos['Cotação'] / todos['P/VP']
        todos['Lucro Líquido'] = (todos['ROE'] /100 ) * 1159400000
        todos['Quantidade de ações'] = 1159400000 / todos['VPA']
        todos['LPA'] = todos['Lucro Líquido'] / todos['Quantidade de ações']
        todos['DPA'] = todos['Cotação'] * (todos['DY'] / 100)
        todos['Payout'] = todos['DPA'] / todos['LPA']
        todos['Expectativa de crescimento'] = (1 - todos['Payout']) * todos['ROE']
        todos['Valuation Bazin'] = todos['DPA'] / 0.06
        todos['Desconto Bazin'] = (todos['Valuation Bazin'] - todos['Cotação']) / todos['Valuation Bazin']
        todos['Valuation Graham'] = (22.5 * todos['LPA'] * todos['VPA']) ** 0.5
        todos['Desconto Graham'] = (todos['Valuation Graham'] - todos['Cotação']) / todos['Valuation Graham']

        todos = todos.fillna(0)

        merged = pd.merge(todos,codigo_nome,on='Código', how="left")
        merged.replace([np.inf, -np.inf], 0, inplace=True)

        #group
        col1, col2, col3 = st.columns([1,1,1])   
        with col1:   
            grouped_setor = merged.groupby('Setor').mean()
            st.dataframe(grouped_setor[['Expectativa de crescimento','Desconto Graham', 'Desconto Bazin']].sort_values('Expectativa de crescimento', ascending=False).reset_index())
        
        with col2:   
            grouped_subsetor = merged.groupby('Subsetor').mean()
            st.dataframe(grouped_subsetor[['Expectativa de crescimento','Desconto Graham', 'Desconto Bazin']].sort_values('Expectativa de crescimento', ascending=False).reset_index())
        
        with col3:   
            grouped_subsetor = merged.groupby('Segmento').mean()
            st.dataframe(grouped_subsetor[['Expectativa de crescimento','Desconto Graham', 'Desconto Bazin']].sort_values('Expectativa de crescimento', ascending=False).reset_index())

        

        #filter
        col1, col2,col3 = st.columns([0.1,0.4,0.1])   
        st.title('')
        
        with col2:   
            st.write('Aplique alguns filtros na tabela')
            exp_min = float(st.number_input(label='Expectativa de crescimento >',value=10,step=5))
            desc_bazin = int(st.number_input(label='Desconto Bazin >',value=0.3,step=0.1))
            desc_graham = float(st.number_input(label='Desconto Graham >',value=0.3,step=0.1))

        filtred = merged.loc[(merged['Expectativa de crescimento']>= exp_min) & (merged['Desconto Bazin']>= desc_bazin) & (merged['Desconto Graham']>= desc_graham) & (merged['Desconto Bazin']!= 0)& (merged['Desconto Graham']!= 0)]
        filtred = filtred.sort_values('Expectativa de crescimento',ascending=False)



        #st.dataframe(merged[['Código','Cotação', 'Pat.Liq', 'Expectativa de crescimento','Desconto Bazin', 'Desconto Graham','Nome',	'Setor','Subsetor',	'Segmento']])
        #AgGrid(filtred[['Código','Cotação', 'Lucro Líquido', 'Expectativa de crescimento','Desconto Bazin', 'Desconto Graham','Nome',	'Setor','Subsetor',	'Segmento']].round(2))
        st.dataframe(filtred[['Código','Cotação', 'Lucro Líquido', 'Expectativa de crescimento','Desconto Bazin', 'Desconto Graham','Nome',	'Setor','Subsetor',	'Segmento']].round(2))




            