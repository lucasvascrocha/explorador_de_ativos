import matplotlib
matplotlib.use('Agg')

#from turtle import width
import streamlit as st
from yahooquery import Ticker
import pandas as pd
import yfinance as yf
# from fbprophet import Prophet
import numpy as np
import plotly.graph_objects as go

import scrap as scraping

import style as style

import html01 as html01



def analise_tecnica_fundamentalista():

    #código para ativar bootstrap css
    st.markdown(
"""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""",unsafe_allow_html=True
    )  
    
    col1, col2,col3 = st.columns([0.1,0.4,0.1])   
    with col2:   
        st.title('Análise Técnica')
        st.subheader('Escolha o ativo que deseja analisar e pressione enter')  
        #nome_do_ativo = st.text_input('Nome do ativo ex: PETR4, VALE3, WEGE3...')
        codigo_nome = pd.read_excel('data/classification_b3.xlsx')
        nome_do_ativo = st.selectbox('Esoolha a ação que deseja analisar', (codigo_nome['TICKER']) )

    style.space(1)

    if nome_do_ativo != "":
        nome_do_ativo = str(nome_do_ativo + '.SA').upper()
        df = Ticker(nome_do_ativo,country='Brazil')
        time = df.history( period='max')

    # ------------------------------ RESUMO ---------------------------- 

        resumo = pd.DataFrame(df.summary_detail)
        resumo = resumo.transpose()
        if len(nome_do_ativo) == 8:
            fundamentus = scraping.get_specific_data(nome_do_ativo[:5])
            fundamentus = pd.DataFrame([fundamentus])
    
            try:

                pfizer = yf.Ticker(nome_do_ativo)
                info = pfizer.info 
                if info['recommendationKey'] == 'buy':
                    recomendation = 'alta'
                else:
                    recomendation = 'baixa'

                #KPIS
                metric1, metric2, metric3, metric4 = st.columns([1,1,1,1])
                
                with metric1:
                    st.metric(label="P/L",value=f"{fundamentus['P/L'][0]}")
                with metric2:
                    st.metric(label="P/VP",value=f"{fundamentus['P/VP'][0]}")
                with metric3:
                    st.metric(label="Tendência", value=f"{recomendation}")
                with metric4:
                    st.metric(label="Próximo dividendo", value=f"{pfizer.calendar.transpose()['Earnings Date'].dt.strftime('%d/%m/%Y')[0]}")

                style.space(2)

                #card info
                html01.card_info(info['longName'], info['sector'], info['industry'], info['longBusinessSummary'], info['website'])

                style.space(2)    

            except:
                exit
            
        else:
            st.write('---------------------------------------------------------------------')
            st.dataframe(resumo) 
            pfizer = yf.Ticker(nome_do_ativo)
            info = pfizer.info 
            st.title('Company Profile')
            st.subheader(info['longName']) 
            try:
                st.markdown('** Sector **: ' + info['sector'])
                st.markdown('** Industry **: ' + info['industry'])
                st.markdown('** Website **: ' + info['website'])
            except:
                exit
        
    # ------------------------------ GRÁFICOS DE RENDIMENTO ---------------------------- 

        if len(nome_do_ativo) == 8:
            
            import datetime
            fundamentalist = df.income_statement()
            fundamentalist['data'] = fundamentalist['asOfDate'].dt.strftime('%d/%m/%Y')
            fundamentalist = fundamentalist.drop_duplicates('asOfDate')
            fundamentalist = fundamentalist.loc[fundamentalist['periodType'] == '12M']

            #volatilidade
            TRADING_DAYS = 360
            returns = np.log(time['close']/time['close'].shift(1))
            returns.fillna(0, inplace=True)
            volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
            vol = pd.DataFrame(volatility.iloc[-360:]).reset_index()

            #sharpe ratio
            sharpe_ratio = returns.mean()/volatility
            sharpe = pd.DataFrame(sharpe_ratio.iloc[-360:]).reset_index()

            div = time.reset_index()
            div['year'] = pd.to_datetime(div['date']).dt.strftime('%Y')
            div_group = div.groupby('year').agg({'close':'mean','dividends':'sum'})
            div_group['dividendo(%)'] = round((div_group['dividends'] * 100 ) / div_group['close'],4)

            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=3, cols=2,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "scatter"}]],
                subplot_titles=("Receita Total","Lucro",'Dividendos (%)','Dividendos unitário R$','Volatilidade', 'Sharpe ratio (Retorno/ Risco)')
            )

            fig.add_trace(go.Bar(x =pfizer.financials.transpose().index,  y=pfizer.financials.transpose()['Total Revenue']), row=1, col=1)

            fig.add_trace(go.Bar(x =pfizer.financials.transpose().index,  y=pfizer.financials.transpose()['Net Income From Continuing Ops']), row=1, col=2)

            fig.add_trace(go.Bar(x =div_group.reset_index().tail(5)['year'],  y=div_group.reset_index().tail(5)['dividendo(%)']),row=2, col=1)

            fig.add_trace(go.Bar(x =div_group.reset_index().tail(5)['year'],  y=div_group.reset_index().tail(5)['dividends']),row=2, col=2)

            fig.add_trace(go.Scatter(x =vol['date'],  y=vol['close']),row=3, col=1)

            fig.add_trace(go.Scatter(x =sharpe['date'],  y=sharpe['close']),row=3, col=2)

            fig.update_layout( height=800, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
            fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

            col1, col2, col3 = st.columns([1,8,1])
            #with col2: 
            st.plotly_chart(fig,use_container_width=True)     
                

        else:
            #volatilidade
            TRADING_DAYS = 160
            returns = np.log(time['close']/time['close'].shift(1))
            returns.fillna(0, inplace=True)
            volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
            vol = pd.DataFrame(volatility.iloc[-160:]).reset_index()

            #sharpe ratio
            sharpe_ratio = returns.mean()/volatility
            sharpe = pd.DataFrame(sharpe_ratio.iloc[-160:]).reset_index()

            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "scatter"}, {"type": "scatter"}]],
                subplot_titles=('Volatilidade', 'Sharpe ratio (Retorno/ Risco)')
            )

            fig.add_trace(go.Scatter(x =vol['date'],  y=vol['close']),row=1, col=1)

            fig.add_trace(go.Scatter(x =sharpe['date'],  y=sharpe['close']),row=1, col=2)

            fig.update_layout( height=800, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
            fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')


            col1, col2, col3 = st.columns([1,8,1])
            #with col2: 
            st.plotly_chart(fig,use_container_width=True)      

    # ------------------------------ GRÁFICOS DE Candlestick---------------------------- 
        with st.expander("Entenda o gráfico de Candlestick, clique para saber mais"):
            st.write("""Gráfico de Candlestick , O formato contém os valores dos preços que a ação atingiu durante o período de tempo que está sendo analisado. São os preços de:""")
            st.write('Abertura: preço pelo qual foi fechado o primeiro negócio do período')
            st.write('Fechamento: preço pelo qual foi fechado o último negócio do período')
            st.write('Máximo: maior preço negociado no período')
            st.write('Mínimo: menor preço negociado no período')   
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, subplot_titles=('Candlestick', 'Volume'), 
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

        fig.update_layout( height=600, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') #width=800 ,
        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')



        st.plotly_chart(fig,use_container_width=True)    
        
    # ------------------------------ GRÁFICOS DE Retorno acumulado---------------------------- 
        with st.expander("Entenda o gráfico de Retorno acumulado, clique para saber mais"):
            st.write("""Gráfico de Retorno acumulado, Acúmulo de retorno calculado desde a data de início escolhida até o dia de hoje.""")

        layout = go.Layout(title="Retorno acumulado",xaxis=dict(title="Data"), yaxis=dict(title="Retorno"))
        fig = go.Figure(layout = layout)
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-365:], y=time.reset_index()['close'][-365:].pct_change().cumsum(), mode='lines', line_width=3,line_color='rgb(0,0,0)'))

        fig.update_layout( height=600, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

        st.plotly_chart(fig,use_container_width=True)   

    # ------------------------------ GRÁFICOS DE Médias móveis---------------------------- 
        with st.expander("Entenda o gráfico de Médias móveis, clique para saber mais"):
            st.write("""Gráfico de Médias móveis, Cada ponto no gráfico representa a média dos últimos x dias, exemplo MM20 = média móvel dos última 20 dias.""")
            st.write("""Com ela, é possível identificar o equilíbrio dos preços no mercado, observando tendências de alta, neutra ou baixa. A representação gráfica das Médias Móveis é normalmente feita por uma linha, que se movimenta conforme os dados novos recebidos para o cálculo.""")

        rolling_200  = time['close'].rolling(window=200)
        rolling_mean_200 = rolling_200.mean()

        rolling_50  = time['close'].rolling(window=72)
        rolling_mean_50 = rolling_50.mean()

        rolling_20  = time['close'].rolling(window=20)
        rolling_mean_20 = rolling_20.mean()

        rolling_10  = time['close'].rolling(window=9)
        rolling_mean_10 = rolling_10.mean()

        layout = go.Layout(title="Médias móveis - ative ou desative clicando na legenda da média",xaxis=dict(title="Data"), yaxis=dict(title="Preço R$"))
        fig = go.Figure(layout = layout)
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=time["close"][-120:], mode='lines', line_width=3,name='Real',line_color='rgb(0,0,0)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=rolling_mean_200[-120:],mode='lines',name='MM(200)',opacity = 0.6))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=rolling_mean_50[-120:],mode='lines',name='MM(72)',opacity = 0.6))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=rolling_mean_20[-120:],mode='lines',name='MM(20)',opacity = 0.6))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=rolling_mean_10[-120:],mode='lines',name='MM(9)',opacity = 0.6,line_color='rgb(100,149,237)'))

        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')
        fig.update_layout(autosize=True, height=600, width=800 ,showlegend=True, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
        

        st.plotly_chart(fig,use_container_width=True)     

    # ------------------------------ GRÁFICOS DE Retração de Fibonacci---------------------------- 
        with st.expander("Entenda o gráfico de Retração de Fibonacci, clique para saber mais"):
            st.write("""Gráfico de Retração de Fibonacci, A retração de Fibonacci é composta por linhas horizontais que cortam a série de preços. A distância entre essas linhas varia obedecendo a chamada série numérica de Fibonacci. Na análise técnica esses valores são expressos em porcentagem e são: 100%, 61,8%, 38,2%, 23,6%, 0%.""")
            st.write("""Muito utilizado para tentar identificar uma tendência de alta ou de baixa. Uma das técnicas é analisar se a ação caiu até a linha de 38,2% (verde) para voltar a subir (retração com tendência de alta) ou se caiu passando os 38,2% pode representar uma tendência de baixa ou alta com pouca força.""")
            st.write('Tente sempre escolher a quantidade de dias analisados para começar a análise após o maior fundo de baixa histórico')
        time_fibo = time.copy()

        
        periodo_fibonacci = int(st.number_input(label='periodo fibonacci - traçada do menor valor encontrado no período de tempo setado abaixo até o maior valor encontrado para frente',value= 45 ))
        
        Price_Min =time_fibo[-periodo_fibonacci:]['low'].min()
        Price_Max =time_fibo[-periodo_fibonacci:]['high'].max()

        Diff = Price_Max-Price_Min
        level1 = Price_Max - 0.236 * Diff
        level2 = Price_Max - 0.382 * Diff
        level3 = Price_Max - 0.618 * Diff
    
        # st.write ('0% >>' f'{round(Price_Max,2)}')
        # st.write ('23,6% >>' f'{round(level1,2)}')
        # st.write ('38,2% >>' f'{round(level2,2)}')
        # st.write ('61,8% >>' f'{round(level3,2)}')
        # st.write ('100% >>' f'{round(Price_Min,2)}')

        time_fibo['Price_Min'] = Price_Min
        time_fibo['level1'] = level1
        time_fibo['level2'] = level2
        time_fibo['level3'] = level3
        time_fibo['Price_Max'] = Price_Max

        layout = go.Layout(title=f'Retração de Fibonacci',xaxis=dict(title="Data"), yaxis=dict(title="Preço"))
        fig = go.Figure(layout = layout)
        fig.add_trace(go.Scatter(x=time_fibo[-periodo_fibonacci:].reset_index()['date'], y=time_fibo[-periodo_fibonacci:].close, mode='lines', line_width=3,name='Preço real',line_color='rgb(0,0,0)'))
        fig.add_trace(go.Scatter(x=time_fibo[-periodo_fibonacci:].reset_index()['date'], y=time_fibo[-periodo_fibonacci:].Price_Min, mode='lines', line_width=0.5,name='100%',line_color='rgb(255,0,0)',))
        fig.add_trace(go.Scatter(x=time_fibo[-periodo_fibonacci:].reset_index()['date'], y=time_fibo[-periodo_fibonacci:].level3, mode='lines', line_width=0.5,name='61,8%',line_color='rgb(255,255,0)',fill= 'tonexty', fillcolor ="rgba(255, 0, 0, 0.2)"))
        fig.add_trace(go.Scatter(x=time_fibo[-periodo_fibonacci:].reset_index()['date'], y=time_fibo[-periodo_fibonacci:].level2, mode='lines', line_width=0.5,name='38,2%',line_color='rgb(0,128,0)',fill= 'tonexty', fillcolor ="rgba(255, 255, 0, 0.2)"))
        fig.add_trace(go.Scatter(x=time_fibo[-periodo_fibonacci:].reset_index()['date'], y=time_fibo[-periodo_fibonacci:].level1, mode='lines', line_width=0.5,name='23,6%',line_color='rgb(128,128,128)',fill= 'tonexty', fillcolor ="rgba(0, 128, 0, 0.2)"))
        fig.add_trace(go.Scatter(x=time_fibo[-periodo_fibonacci:].reset_index()['date'], y=time_fibo[-periodo_fibonacci:].Price_Max, mode='lines', line_width=0.5,name='0%',line_color='rgb(0,0,255)',fill= 'tonexty', fillcolor ="rgba(128, 128, 128, 0.2)"))

        fig.update_layout( height=600, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

        st.plotly_chart(fig,use_container_width=True)       

    # ------------------------------ GRÁFICOS DE RSI---------------------------- 
        try:
            
            delta = time['close'][-periodo_RSI:].diff()
            up, down = delta.copy(), delta.copy()

            up[up < 0] = 0
            down[down > 0] = 0

            period = 14
                
            rUp = up.ewm(com=period - 1,  adjust=False).mean()
            rDown = down.ewm(com=period - 1, adjust=False).mean().abs()

            time['RSI_' + str(period)] = 100 - 100 / (1 + rUp / rDown)
            time['RSI_' + str(period)].fillna(0, inplace=True)

            layout = go.Layout(title=f'RSI {periodo_RSI}',xaxis=dict(title="Data"), yaxis=dict(title="%RSI"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_RSI:], y=round(time['RSI_14'][-periodo_RSI:],2), mode='lines', line_width=3,name=f'RSI {periodo_RSI}',line_color='rgb(0,0,0)'))

            fig.update_layout( height=600, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
            fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

            periodo_RSI = int(st.number_input(label='periodo RSI',value=90))

            with st.expander("Entenda o gráfico de Índice de Força Relativa (RSI), clique para saber mais"):
                st.write("""Gráfico de Índice de Força Relativa (RSI), é um indicador versátil que mede a velocidade e a mudança do movimento dos preços. O RSI pode ser usado para: Determine quando um instrumento está sobrecomprado ou sobrevendido.""")  
                st.write('Comprar ativos que apresentem um baixo valor de RSI pode ser um bom indicador de desconto!')

            st.plotly_chart(fig,use_container_width=True)     

        except:
            exit

    # ------------------------------ GRÁFICOS DE pivôs---------------------------- 
        with st.expander("Entenda o gráfico de pivôs, clique para saber mais"):
                st.write("""O Pivot Point pode ser utilizado para calcular as possíveis zonas de suporte e resistência do ativo para o período desejado, o que ajuda a entender a pressão compradora (suporte) e a pressão vendedora (resistência).""") 
                st.write(' Os pontos do pivô podem ser utilizados como valores de stop-loss (preço de saída com perda) ou stop-gain (preço de saída com lucro) por exemplo.') 

                

        periodo_pivo = int(st.number_input(label='periodo pivô',value=20))

        time['PP'] = pd.Series((time['high'] + time['low'] + time['close']) /3)  
        time['R1'] = pd.Series(2 * time['PP'] - time['low'])  
        time['S1'] = pd.Series(2 * time['PP'] - time['high'])  
        time['R2'] = pd.Series(time['PP'] + time['high'] - time['low'])  
        time['S2'] = pd.Series(time['PP'] - time['high'] + time['low']) 

        layout = go.Layout(title=f'Pivô',xaxis=dict(title="Data"), yaxis=dict(title="Preço"))
        fig = go.Figure(layout = layout)
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_pivo:], y=round(time['close'][-periodo_pivo:],2), mode='lines', line_width=3,name=f'preço real',line_color='rgb(0,0,0)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_pivo:], y=round(time['PP'][-periodo_pivo:],2), mode='lines', line_width=1,name=f'Ponto do pivô',line_color='rgb(0,128,0)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_pivo:], y=round(time['R1'][-periodo_pivo:],2), mode='lines', line_width=1,name=f'Resistência 1',line_color='rgb(100,149,237)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_pivo:], y=round(time['S1'][-periodo_pivo:],2), mode='lines', line_width=1,name=f'Suporte 1',line_color='rgb(100,149,237)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_pivo:], y=round(time['R2'][-periodo_pivo:],2), mode='lines', line_width=1,name=f'Resistência 2',line_color='rgb(255,0,0)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_pivo:], y=round(time['S2'][-periodo_pivo:],2), mode='lines', line_width=1,name=f'Suporte 2',line_color='rgb(255,0,0)'))

        fig.update_layout( height=600, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

        st.plotly_chart(fig,use_container_width=True)       

    # ------------------------------ GRÁFICOS DE Bolinger---------------------------- 
        with st.expander("Entenda o gráfico de Bolinger, clique para saber mais"):
            st.write("""Quando o preço do ativo ultrapassa a banda superior, observamos uma tendência de alta do ativo. Por outro lado, se o preço fica abaixo da banda inferior, há então uma tendência de baixa. Entretanto, deve-se ficar atento aos sinais de força dos ativos ao ultrapassar as bandas.""") 

        periodo_bolinger = int(st.number_input(label='periodo Bolinger',value=180))

        time['MA20'] = time['close'].rolling(20).mean()
        time['20 Day STD'] = time['close'].rolling(window=20).std()
        time['Upper Band'] = time['MA20'] + (time['20 Day STD'] * 2)
        time['Lower Band'] = time['MA20'] - (time['20 Day STD'] * 2)

        layout = go.Layout(title=f'Banda de Bolinger',xaxis=dict(title="Data"), yaxis=dict(title="Preço"))
        fig = go.Figure(layout = layout)
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_bolinger:], y=round(time['Upper Band'][-periodo_bolinger:],2), mode='lines', line_width=1,name=f'Banda superior',line_color='rgb(255,0,0)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_bolinger:], y=round(time['Lower Band'][-periodo_bolinger:],2), mode='lines', line_width=1,name=f'Banda inferior',line_color='rgb(255,0,0)',fill= 'tonexty', fillcolor ="rgba(255, 0, 0, 0.1)",opacity=0.2))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_bolinger:], y=round(time['close'][-periodo_bolinger:],2), mode='lines', line_width=3,name=f'preço real',line_color='rgb(0,0,0)'))
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-periodo_bolinger:], y=round(time['MA20'][-periodo_bolinger:],2), mode='lines', line_width=2,name=f'MM 20',line_color='rgb(0,128,0)'))
  
        fig.update_layout( height=600, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

        st.plotly_chart(fig,use_container_width=True)   


    # # ------------------------------ Previsões---------------------------- 
    #     with st.expander("Entenda o gráfico de Previsão, clique para saber mais"):
    #        st.write("""Previsão que leva em conta apenas o movimento gráfico histórico do ativo, porém sabemos que o preço do ativo varia por outros diversos fatores. Com isso, esse é um parâmetro apenas como expectativa caso todos os outros fatores envolvidos se mantivessem.""") 
   

    #     st.subheader('Previsões')

    #     st.write('As previsões são feitas levando em conta apenas o movimento gráfico, porém o movimento do preço de um ativo é influenciado por diversos outros fatores, com isso, deve se considerar as previsões como uma hipótese de o preço do ativo variar somente pela sua variação gráfica')

    #     st.write('Previsão considerando os últimos 365 dias, pode ser entendida como uma tendência dos dados segundo o último ano')
        
    #     st.write('Opção de alterar a previsão: caso esteja buscando resultados a curto prazo é possível alterar o "periodo analisado" para fazer previsões apenas com base nos últimos x dias. Neste caso o movimento gráfico para trás dos dias selecionados não serão levados em conta')
    #     periodo_analisado = int(st.number_input(label='período analisado (dias de resultados passados)',value=360))

    #     st.write('Opção de alterar a previsão: possibilidade de prever resultados futuros por mais de 30 dias')
    #     periodo_futuro = int(st.number_input(label='período futuro a prever (dias)',value=30))

    #     time = time.reset_index()
    #     time = time[['date','close']]
    #     time.columns = ['ds','y']

    #     #Modelling
    #     m = Prophet()
    #     m.fit(time[-periodo_analisado:])
    #     future = m.make_future_dataframe(periods= periodo_futuro, freq='B')
    #     forecast = m.predict(future[-periodo_futuro:])

    #     from fbprophet.plot import plot_plotly, plot_components_plotly

    #     fig1 = plot_plotly(m, forecast)
    #     fig1.update_layout( height=600, width=800 ,showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
    #     fig1.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

    #     st.plotly_chart(fig1,use_container_width=True)    

    #     st.subheader('Tendência diária e semanal')
    #     st.write('0 = segunda, 1 = terça, ... , 5 = sábado, 6 = domingo')
    #     fig2 = m.plot_components(forecast,uncertainty = False,weekly_start=1)
        

    #     st.plotly_chart(fig2,use_container_width=True)    
        





















        #st.write('Previsão considerando as últimas semanas, pode ser entendida como uma tendência dos dados segundo os últimos dias. Leva em consideração diversos fatores como: Índice de força relativa RSI, oscilador estocástico %K, Indicador Willian %R além do movimento gráfico dos últimos dias')

        #predict = stocker.predict.tomorrow(nome_do_ativo)

        #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação feche no valor de: R$',f'{predict[0]}')

        #preço_ontem= round(time['y'][-1:].values[0],2)
        #if predict[0] < preço_ontem:
            #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação caia de ',f'{preço_ontem}', 'para valor de: R$ ',f'{predict[0]}')
        #else:
            #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação suba de ',f'{preço_ontem}', 'para valor de: R$ ',f'{predict[0]}')
                        