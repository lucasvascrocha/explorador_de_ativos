import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import datetime as dt 
from yahooquery import Ticker
import plotly.graph_objects as go

from collections import OrderedDict

import matplotlib

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

def rastreador():
        #código para ativar bootstrap css
        st.markdown(
        """
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        """,unsafe_allow_html=True
        )  
        
        col1, col2,col3 = st.columns([0.1,0.4,0.1])   
        with col2:   
          
          st.title('Rastreador de oportunidades por setup')
          
          st.expander('Este rastreador identifica oportunidades para swing trade vasculhando as principais ações listadas na B3, o filtro consiste em encontrar ativos que tenham médias móveis exponenciais de 9 e 72 cruzadas para cima')
          
          st.subheader('Aplique alguns filtros para acelerar o rastreador')
          st.write('Serão rastreadas apenas as ações listadas na tabela filtrada abaixo')

          PVP_máximo = float(st.number_input(label='PVP máximo',value=1.0))
          Patr_liq_min = int(st.number_input(label='Patrimônio líquido mínimo',value=1000000000))
          cotacao_max = float(st.number_input(label='Cotação máxima',value=100))
          

          lista = scraping.get_data()
          todos = pd.DataFrame(flatten(lista).keys()).transpose()
          todos.columns = todos.iloc[0]
          for i in range(len(lista)):
            todos = pd.concat([todos,pd.DataFrame(lista[i]).transpose()])
          
          todos = todos.iloc[1:]
          todos = todos.replace(',','.', regex=True)
          todos = todos.apply(pd.to_numeric,errors='ignore').round(2)
          todos['Pat.Liq'] = todos['Pat.Liq'].str.replace(r"[^0-9]+", '').replace('.','').replace(',','').replace('-','').astype(float)

          todos = todos.loc[(todos['P/VP']<= PVP_máximo) & (todos['Pat.Liq']>= Patr_liq_min) & (todos['cotacao']<= cotacao_max)]
          show = todos.reset_index()
        st.dataframe(show)

        st.subheader('Setup')
        col1, col2,col3 = st.columns([0.1,0.4,0.1]) 
        with col2:  
          st.checkbox('Médias móveis exponenciais de 9 e 72 cruzadas para cima')




        if st.button("Iniciar rastreador"):

          # lista = scraping.get_data()
          # todos = pd.DataFrame(flatten(lista).keys()).transpose()
          # todos.columns = todos.iloc[0]
                    
          # for i in range(len(lista)):
          #   todos = pd.concat([todos,pd.DataFrame(lista[i]).transpose()])
          
          # todos = todos.iloc[1:]
          # todos = todos.replace(',','.', regex=True)
          # todos = todos.apply(pd.to_numeric,errors='ignore').round(2)
          # todos['Pat.Liq'] = todos['Pat.Liq'].str.replace(r"[^0-9]+", '').replace('.','').replace(',','').replace('-','').astype(float)
          # todos = todos.loc[(todos['P/VP']<= PVP_máximo) & (todos['Pat.Liq']>= Patr_liq_min) & (todos['cotacao']<= cotacao_max)]
          
          
          start = (dt.datetime.today() + dt.timedelta(days=-300)).strftime(format='20%y-%m-%d')
          dia_limite = (dt.datetime.today() + dt.timedelta(days=-30)).strftime(format='20%y-%m-%d')


          with st.expander("Aguarde estamos vasculhando todas as ações da bolsa (Mantenha esta barra minimizada)!"):
              save = []
              #for i in range(len(tudo)):
              for i in range(len(todos)):
                try:

                  #nome_do_ativo = str(tudo.iloc[i][0] + '.SA')
                  nome_do_ativo = str(todos.index[i] + '.SA')
                  #filtra todos que cruzaram média nos últimos 50 dias pelo menos
                  try:
                    df = Ticker(nome_do_ativo ,country='Brazil')
                    time = df.history( start= start )
                    rolling_9  = time['close'].rolling(window=9)
                    rolling_mean_9 = rolling_9.mean().round(1)

                    rolling_72  = time['close'].rolling(window=72)
                    rolling_mean_72 = rolling_72.mean().round(1)
                    time['MM9'] = rolling_mean_9.fillna(0)
                    time['MM72'] = rolling_mean_72.fillna(0)
                    time['cruzamento'] =  time['MM9'] - time['MM72']
                    buy = time.tail(50).loc[(time.tail(50)['cruzamento']==0)]
                  except:
                    exit


                except:
                  exit


                if buy.empty == False:
                  try:
                    #filtra todo mundo que tem a MM 72 > que a MM 9 e quem tem volume do último dia > 5000  
                    if time['MM72'].iloc[-1] < time['MM9'].iloc[-1] and  time.tail(1)['volume'][0] > 10000:
                      save.append(buy.index[0][0])
                      print(buy.index[0][0])
                      #layout = go.Layout(title="Resultados",xaxis=dict(title="Data"), yaxis=dict(title="Preço R$"))
                      #fig = go.Figure(layout = layout)
                      #fig.add_trace(go.Candlestick(x=time.reset_index()['date'][-50:], open=time['open'][-50:],high=time['high'][-50:],low=time['low'][-50:],close=time['close'][-50:]))
                      #fig.update_layout(autosize=False,width=1000,height=800,)
                      #fig.show()
                      #print()
                    else:
                      continue
                  except:         
                    exit

                else:
                  exit
            
              
          st.dataframe(save)
          save = pd.DataFrame(save)
          
          from plotly.subplots import make_subplots
          
          
          for i in range(len(save)):
              df = Ticker(save.iloc[i] ,country='Brazil')
              time = df.history( start= start )



              fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
              vertical_spacing=0.03, subplot_titles=(st.write(save.iloc[i]), 'Volume'), 
              row_width=[0.2, 0.7])

              # Plot OHLC on 1st row
              fig.add_trace(go.Candlestick(x=time.reset_index()['date'][-90:],
                          open=time['open'][-90:], high=time['high'][-90:],
                          low=time['low'][-90:], close=time['close'][-90:], name="OHLC"), 
                          row=1, col=1)            

              # Bar trace for volumes on 2nd row without legend
              fig.add_trace(go.Bar(x=time.reset_index()['date'][-90:], y=time['volume'][-90:], showlegend=False), row=2, col=1)

              # Do not show OHLC's rangeslider plot 
              fig.update(layout_xaxis_rangeslider_visible=False)
              #fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
              fig.update_layout(height=600, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') #width=800 ,
              fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

              st.plotly_chart(fig,use_container_width=True)    