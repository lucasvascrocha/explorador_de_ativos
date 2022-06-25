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
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            #st.image('https://media.giphy.com/media/d83YIjgW4uyTpYfjbd/giphy.gif', width=400)
            st.write("")
        with col3:
            st.write("")

        #st.image('https://media.giphy.com/media/d83YIjgW4uyTpYfjbd/giphy.gif', width=400)    
        #image = Image.open('imagens/logo.jpg')
        #st.image(image, use_column_width=True)  
        
        lista = scraping.get_data()
        todos = pd.DataFrame(flatten(lista).keys()).transpose()
        todos.columns = todos.iloc[0]
        
        for i in range(len(lista)):
          todos = pd.concat([todos,pd.DataFrame(lista[i]).transpose()])
        
        todos = todos.iloc[1:]
        
        
        start = (dt.datetime.today() + dt.timedelta(days=-300)).strftime(format='20%y-%m-%d')
        dia_limite = (dt.datetime.today() + dt.timedelta(days=-30)).strftime(format='20%y-%m-%d')
        
        st.title('Rastreador de trade')
        
        st.write('Este rastreador identifica oportunidades para swing trade vasculhando as principais ações listadas na B3, o filtro consiste em encontrar ativos que tenham médias móveis exponenciais de 9 e 72 cruzadas para cima')
        
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
                  if time['MM72'].iloc[-1] < time['MM9'].iloc[-1] and  time.tail(1)['volume'][0] > 5000:
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
            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig)