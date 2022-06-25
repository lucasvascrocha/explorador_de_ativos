import matplotlib
matplotlib.use('Agg')

#from turtle import width
import streamlit as st
from yahooquery import Ticker
import pandas as pd
import yfinance as yf
#from fbprophet import Prophet
import numpy as np
import plotly.graph_objects as go

import scrap as scraping

import style as style

def analise_tecnica_fundamentalista2():



    st.markdown(
"""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""",unsafe_allow_html=True
    )  
    
    col1, col2, col3 = st.columns([2,6,2])

    #with col2:                   
    st.title('Análise Técnica e fundamentalista')
    st.subheader('Escolha o ativo que deseja analisar e pressione enter')
    nome_do_ativo = st.text_input('Nome do ativo')
    st.title('')


    #st.write('Este explorador funciona melhor para ações, porém também suporta alguns fundos imobiliários')    
    #st.write('Os parâmetros utilizados em grande maioria foram seguindo as teorias de Benjamin Graham')

    if nome_do_ativo != "":
        nome_do_ativo = str(nome_do_ativo + '.SA')
        #st.subheader('Analisando os dados')
        df = Ticker(nome_do_ativo,country='Brazil')
        time = df.history( period='max')
        #st.dataframe(time.tail())

    # ------------------------------ RESUMO ---------------------------- 

        resumo = pd.DataFrame(df.summary_detail)
        resumo = resumo.transpose()
        if len(nome_do_ativo) == 8:
            fundamentus = scraping.get_specific_data(nome_do_ativo[:5])
            fundamentus = pd.DataFrame([fundamentus])
            
            try:


                pfizer = yf.Ticker(nome_do_ativo)
                info = pfizer.info 

                # st.markdown('** Próximo pagamento de dividendo: **: ' + (pfizer.calendar.transpose()['Earnings Date'].dt.strftime('%d/%m/%Y')[0]))

                #KPIS
                # metric1, metric2, metric3 = st.columns([1,1,1])
                #st.metric(label="Temperature",value=f"{fundamentus['P/L'][0]}")
                # with metric1:
                #     st.metric(label="Temperature",value=f"{fundamentus['P/L'][0]}")
                # with metric2:
                #     st.metric(label="Temperature",value=f"{fundamentus['P/VP'][0]}")
                # with metric3:
                #     st.metric(label="Temperature", value=f"{info['recommendationKey']}")
                
                # metric2.metric(f'P/VP: {fundamentus['P/VP'][0]}')
                # metric3.metric(f'{info['recommendationKey']}')



                st.markdown(
                        f"""
                        <div class="card-deck" style= "-webkit-box-orient: horizontal;  width: 830px;" >

                        <div class="card">
                            <div class="card-body text-center">
                            <p class="card-text" style="font-size: 20px; color: rgb(167, 174, 177); font-weight: bold;">P/L</p>
                            <p class="card-text" style=" font-size: 40px">{fundamentus['P/L'][0]}</p>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-body text-center">
                            <p class="card-text" style="font-size: 20px; color: rgb(167, 174, 177); font-weight: bold;">P/VP</p>
                            <p class="card-text" style=" font-size: 40px;">{fundamentus['P/VP'][0]}</p>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-body text-center">
                            <p class="card-text" style="font-size: 20px;color: rgb(167, 174, 177); font-weight: bold; ">Recomendação</p>
                            <p class="card-text" style=" font-size: 40px; ">{info['recommendationKey']}</p>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-body text-center">
                            <p class="card-text" style="font-size: 20px;color: rgb(167, 174, 177); font-weight: bold;">Próximo dividendo</p>
                            <p class="card-text" style=" font-size: 20px;">{pfizer.calendar.transpose()['Earnings Date'].dt.strftime('%d/%m/%Y')[0]}</p>
                            </div>
                        </div>
                        </div>
                        """,unsafe_allow_html=True
                )

                st.header('')  

                #card 
                st.markdown(
                f"""
            <div class="card-main" style="width: 50rem;" "border-radius: 30px;">
            <div class="card-body">
                <h5 class="card-title" style="text-align: center;">{info['longName']}</h5>
                <h6 class="card-subtitle mb-2 text-muted" style="text-align: center;">{info['sector']}</h6>
                <h6 class="card-subtitle mb-2 text-muted" style="text-align: center;">{info['industry']}</h6>
                <p class="card-text" style="text-align: left;">{info['longBusinessSummary']}</p>
                <a href="#" class="card-link" style="text-align: center;">{info['website']}</a>
            </div>
            </div>
                """, unsafe_allow_html=True
                )       

                st.header('')       

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

            fig.update_layout(height=800, width=800, showlegend=False)#, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

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

            fig.update_layout(height=800, showlegend=False) #, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

    # ------------------------------ GRÁFICOS DE Candlestick---------------------------- 
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
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
        fig.update_layout(autosize=False,width=800,height=800 )#, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
        
    # ------------------------------ GRÁFICOS DE Retorno acumulado---------------------------- 

        layout = go.Layout(title="Retorno acumulado",xaxis=dict(title="Data"), yaxis=dict(title="Retorno"))
        fig = go.Figure(layout = layout)
        fig.add_trace(go.Scatter(x=time.reset_index()['date'][-365:], y=time.reset_index()['close'][-365:].pct_change().cumsum(), mode='lines', line_width=3,line_color='rgb(0,0,0)'))
        fig.update_layout(autosize=False,width=800,height=800) #, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

    # ------------------------------ GRÁFICOS DE Médias móveis---------------------------- 
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
        # fig.add_trace(go.Candlestick(x=time.reset_index()['date'][-120:], open=time['open'][-120:],high=time['high'][-120:],low=time['low'][-120:],close=time['close'][-120:]))
        fig.update_layout(autosize=False,width=800,height=800) #, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

    # ------------------------------ GRÁFICOS DE Retração de Fibonacci---------------------------- 

        time_fibo = time.copy()

        
        periodo_fibonacci = int(st.number_input(label='periodo fibonacci - traçada do menor valor encontrado no período de tempo setado abaixo até o maior valor encontrado para frente',value= 45 ))
        
        Price_Min =time_fibo[-periodo_fibonacci:]['low'].min()
        Price_Max =time_fibo[-periodo_fibonacci:]['high'].max()

        Diff = Price_Max-Price_Min
        level1 = Price_Max - 0.236 * Diff
        level2 = Price_Max - 0.382 * Diff
        level3 = Price_Max - 0.618 * Diff
        
        st.write ('0% >>' f'{round(Price_Max,2)}')
        st.write ('23,6% >>' f'{round(level1,2)}')
        st.write ('38,2% >>' f'{round(level2,2)}')
        st.write ('61,8% >>' f'{round(level3,2)}')
        st.write ('100% >>' f'{round(Price_Min,2)}')

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
        fig.update_layout(autosize=False,width=800,height=800) #, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

    # ------------------------------ GRÁFICOS DE RSI---------------------------- 

        periodo_RSI = int(st.number_input(label='periodo RSI',value=90))

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

        fig.update_layout(autosize=False,width=800,height=800) #, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

    # ------------------------------ GRÁFICOS DE pivôs---------------------------- 

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
        fig.update_layout(autosize=False,width=800,height=800) #, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

    # ------------------------------ GRÁFICOS DE Bolinger---------------------------- 

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
        fig.update_layout(autosize=False,width=800,height=800) #, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

    # ------------------------------ Previsões---------------------------- 

        # st.subheader('Previsões')

        # st.write('As previsões são feitas levando em conta apenas o movimento gráfico, porém o movimento do preço de um ativo é influenciado por diversos outros fatores, com isso, deve se considerar as previsões como uma hipótese de o preço do ativo variar somente pela sua variação gráfica')

        # st.write('Previsão considerando os últimos 365 dias, pode ser entendida como uma tendência dos dados segundo o último ano')
        
        # st.write('Opção de alterar a previsão: caso esteja buscando resultados a curto prazo é possível alterar o "periodo analisado" para fazer previsões apenas com base nos últimos x dias. Neste caso o movimento gráfico para trás dos dias selecionados não serão levados em conta')
        # periodo_analisado = int(st.number_input(label='período analisado (dias de resultados passados)',value=360))

        # st.write('Opção de alterar a previsão: possibilidade de prever resultados futuros por mais de 30 dias')
        # periodo_futuro = int(st.number_input(label='período futuro a prever (dias)',value=30))

        # time = time.reset_index()
        # time = time[['date','close']]
        # time.columns = ['ds','y']

        # #Modelling
        # m = Prophet()
        # #m.fit(time[-360:])
        # m.fit(time[-periodo_analisado:])
        # #future = m.make_future_dataframe(periods=30, freq='B')
        # future = m.make_future_dataframe(periods= periodo_futuro, freq='B')
        # #forecast = m.predict(future[-30:])
        # forecast = m.predict(future[-periodo_futuro:])

        # from fbprophet.plot import plot_plotly, plot_components_plotly

        # fig1 = plot_plotly(m, forecast)
        # fig1.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        # st.plotly_chart(fig1)

        # st.subheader('Tendência diária e semanal')
        # st.write('0 = segunda, 1 = terça, ... , 5 = sábado, 6 = domingo')
        # fig2 = m.plot_components(forecast,uncertainty = False,weekly_start=1)
        
        # st.plotly_chart(fig2)
        

        #st.write('Previsão considerando as últimas semanas, pode ser entendida como uma tendência dos dados segundo os últimos dias. Leva em consideração diversos fatores como: Índice de força relativa RSI, oscilador estocástico %K, Indicador Willian %R além do movimento gráfico dos últimos dias')

        #predict = stocker.predict.tomorrow(nome_do_ativo)

        #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação feche no valor de: R$',f'{predict[0]}')

        #preço_ontem= round(time['y'][-1:].values[0],2)
        #if predict[0] < preço_ontem:
            #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação caia de ',f'{preço_ontem}', 'para valor de: R$ ',f'{predict[0]}')
        #else:
            #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação suba de ',f'{preço_ontem}', 'para valor de: R$ ',f'{predict[0]}')
                        