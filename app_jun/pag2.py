from multiprocessing import Value
import matplotlib
matplotlib.use('Agg')

import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt 
from collections import OrderedDict

import style as style



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


def comparacao_ativos():
        #código para ativar bootstrap css
        st.markdown(
        """
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        """,unsafe_allow_html=True
        )  
        
        col1, col2,col3 = st.columns([0.1,0.4,0.1])   
        with col2:   
                st.title('Comparação de ativos')
                st.subheader('Escolha 4 ativos para comparar')
                codigo_nome = pd.read_excel('data/classification_b3.xlsx')
                nome_do_ativo1 = st.selectbox('Nome do 1º ativo', (codigo_nome['TICKER']),key=1 )
                nome_do_ativo2 = st.selectbox('Nome do 2º ativo', (codigo_nome['TICKER']),key=2)
                nome_do_ativo3 = st.selectbox('Nome do 3º ativo', (codigo_nome['TICKER']),key=3 )
                nome_do_ativo4 = st.selectbox('Nome do 4º ativo', (codigo_nome['TICKER']),key=4 )
                style.space(1)
                
        if nome_do_ativo4 != "":
                st.subheader('Analisando os dados')
                nome_do_ativo1 = str(nome_do_ativo1 + '.SA').upper()
                nome_do_ativo2 = str(nome_do_ativo2 + '.SA').upper()
                nome_do_ativo3 = str(nome_do_ativo3 + '.SA').upper()
                nome_do_ativo4 = str(nome_do_ativo4 + '.SA').upper()
                
                df = Ticker([nome_do_ativo1,nome_do_ativo2,nome_do_ativo3,nome_do_ativo4],country='Brazil')
                time = df.history( start='2020-01-01', end = (dt.datetime.today() + dt.timedelta(days=1)).strftime(format='20%y-%m-%d'))
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
                todos = todos.apply(pd.to_numeric,errors='ignore').round(2)
                todos.rename(columns={'cotacao': 'Cotação'}, inplace=True)
                comparar = todos.loc[todos.index.isin([nome_do_ativo1[:5],nome_do_ativo2[:5],nome_do_ativo3[:5],nome_do_ativo4[:5]])]
                
                st.dataframe(comparar)
                # st.dataframe(comparar.style.format({"Cotação": "{:.2f}", "P/L": "{:.2f}", "P/VP": "{:.2f}", "P/Ativo": "{:.2f}"
                # , "P/EBIT": "{:.2f}", "P/Ativ.Circ.Liq.": "{:.2f}", "EBITDA": "{:.2f}", "Liq.Corr.": "{:.2f}", "Liq.2m.": "{:.2f}"
                # , "Pat.Liq": "{:.2f}", "Div.Brut/Pat.": "{:.2f}"                
                #       }))


        # ------------------------------ INÍCIO Comparação DY ---------------

                col1, col2 = st.columns([0.5,0.5])   
                with col1:   
                        layout = go.Layout(title="DY",xaxis=dict(title="Ativo"), yaxis=dict(title="DY %"))
                        fig = go.Figure(layout = layout)
                        fig.add_trace(go.Bar(x=comparar.sort_values('DY',ascending=True).index, y=comparar.sort_values('DY',ascending=True)['DY'] ))
                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                        st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ INÍCIO Comparação P/L ---------------
                with col2: 
                        layout = go.Layout(title="P/L",xaxis=dict(title="Ativo"), yaxis=dict(title="P/L"))
                        fig = go.Figure(layout = layout)
                        fig.add_trace(go.Bar(x=comparar.sort_values('P/L',ascending=True).index, y=comparar.sort_values('P/L',ascending=True)['P/L'] ))
                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                        st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ INÍCIO Comparação P/V---------------
                with col1:  
                        layout = go.Layout(title="P/VP",xaxis=dict(title="Ativo"), yaxis=dict(title="P/VP"))
                        fig = go.Figure(layout = layout)
                        fig.add_trace(go.Bar(x=comparar.sort_values('P/VP',ascending=True).index, y=comparar.sort_values('P/VP',ascending=True)['P/VP'] ))
                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                        st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ INÍCIO Comparação P/L * P/VP---------------

                with col2: 
                        layout = go.Layout(title="P/L X P/VP",xaxis=dict(title="Ativo"), yaxis=dict(title="P/L X P/VP"))
                        fig = go.Figure(layout = layout)
                        fig.add_trace(go.Bar(x=comparar.index, y=comparar['P/L'] * comparar['P/VP'] ))
                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                        st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ GRÁFICOS DE retorno acumulado---------------------------- 

                periodo_inicio = int(st.number_input(label='periodo retorno acumulado',value=360))

                ret = time.reset_index()
                layout = go.Layout(title="Retorno acumulado",xaxis=dict(title="Data"), yaxis=dict(title="Retorno"))
                fig = go.Figure(layout = layout)
                for i in range(len(ret['symbol'].unique())):
                        fig.add_trace(go.Scatter(x=ret.loc[ret['symbol']==ret['symbol'].unique()[i]][-periodo_inicio:]['date'], y=ret.loc[ret['symbol']==ret['symbol'].unique()[i]][-periodo_inicio:]['close'].pct_change().cumsum(),mode='lines',name=ret.reset_index()['symbol'].unique()[i]))

                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ GRÁFICOS DE MÉDIAS MÓVEIS 50---------------------------- 

                rolling_50  = time['close'].rolling(window=50)
                rolling_mean_50 = rolling_50.mean()
                rolling_mean_50 = pd.DataFrame(rolling_mean_50.reset_index())
                # mm50 = time.reset_index()


                layout = go.Layout(title="MÉDIAS MÓVEIS 50",xaxis=dict(title="Data"), yaxis=dict(title="Preço R$"))
                fig = go.Figure(layout = layout)
                for i in range(len(rolling_mean_50['symbol'].unique())):
                        fig.add_trace(go.Scatter(x=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['date'], y=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['close'],mode='lines',name=time.reset_index()['symbol'].unique()[i]))

                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ GRÁFICOS DE MÉDIAS MÓVEIS 20---------------------------- 

                rolling_50  = time['close'].rolling(window=20)
                rolling_mean_50 = rolling_50.mean()
                rolling_mean_50 = pd.DataFrame(rolling_mean_50.reset_index())
                # mm50 = time.reset_index()


                layout = go.Layout(title="MÉDIAS MÓVEIS 20",xaxis=dict(title="Data"), yaxis=dict(title="Preço R$"))
                fig = go.Figure(layout = layout)
                for i in range(len(rolling_mean_50['symbol'].unique())):
                        fig.add_trace(go.Scatter(x=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['date'], y=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['close'],mode='lines',name=time.reset_index()['symbol'].unique()[i]))

                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ GRÁFICOS DE volatilidade--------------------------- 
                col1, col2 = st.columns([0.5,0.5])   
                with col1:
                        TRADING_DAYS = 360
                        returns = np.log(time['close']/time['close'].shift(1))
                        returns.fillna(0, inplace=True)
                        volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
                        vol = pd.DataFrame(volatility).reset_index()
                        vol = vol.dropna()

                        layout = go.Layout(title=f"Volatilidade",xaxis=dict(title="Data"), yaxis=dict(title="Volatilidade"))
                        fig = go.Figure(layout = layout)
                        for i in range(len(vol['symbol'].unique())):
                                fig.add_trace(go.Scatter(x=vol.loc[vol['symbol']==vol['symbol'].unique()[i]]['date'], y=vol.loc[vol['symbol']==vol['symbol'].unique()[i]]['close'],name=vol['symbol'].unique()[i] ))
                                fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                                fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                        st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ GRÁFICOS DE sharpe_ratio--------------------------- 
                with col2:
                        sharpe_ratio = returns.mean()/volatility
                        sharpe = pd.DataFrame(sharpe_ratio).reset_index()
                        sharpe = sharpe.dropna()

                        layout = go.Layout(title=f"SHARP (Risco / Volatilidade)",xaxis=dict(title="Data"), yaxis=dict(title="Sharp"))
                        fig = go.Figure(layout = layout)
                        for i in range(len(sharpe['symbol'].unique())):
                                fig.add_trace(go.Scatter(x=sharpe.loc[sharpe['symbol']==sharpe['symbol'].unique()[i]]['date'], y=sharpe.loc[sharpe['symbol']==sharpe['symbol'].unique()[i]]['close'],name=sharpe['symbol'].unique()[i] ))
                                fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                                fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                        st.plotly_chart(fig,use_container_width=True)   

        # ------------------------------ GRÁFICOS DE correlação-------------------------- 
                try:

                        time = time.reset_index()
                        time = time[['symbol','date','close']]
                        df_1 = time.loc[time['symbol'] == time['symbol'].unique()[0]]
                        df_1 = df_1.set_index('date')
                        df_1.columns = df_1.columns.values + '-' + df_1.symbol.unique() 
                        df_1.drop(df_1.columns[0],axis=1,inplace=True)
                        df_2 = time.loc[time['symbol'] == time['symbol'].unique()[1]]
                        df_2 = df_2.set_index('date')
                        df_2.columns = df_2.columns.values + '-' + df_2.symbol.unique() 
                        df_2.drop(df_2.columns[0],axis=1,inplace=True)
                        df_3 = time.loc[time['symbol'] == time['symbol'].unique()[2]]
                        df_3 = df_3.set_index('date')
                        df_3.columns = df_3.columns.values + '-' + df_3.symbol.unique() 
                        df_3.drop(df_3.columns[0],axis=1,inplace=True)
                        df_4 = time.loc[time['symbol'] == time['symbol'].unique()[3]]
                        df_4 = df_4.set_index('date')
                        df_4.columns = df_4.columns.values + '-' + df_4.symbol.unique() 
                        df_4.drop(df_4.columns[0],axis=1,inplace=True)

                        merged = pd.merge(pd.merge(pd.merge(df_1,df_2,left_on=df_1.index,right_on=df_2.index,how='left'),df_3,left_on='key_0',right_on=df_3.index,how='left'),df_4,left_on='key_0',right_on=df_4.index,how='left').rename({'key_0':'date'},axis=1).set_index('date')

                        retscomp = merged.pct_change()

                
                        #plt.figure(figsize=(10,8))

                        #sns.heatmap(retscomp.corr(),annot=True)
                        #st.set_option('deprecation.showPyplotGlobalUse', False)
                        #st.pyplot()

                        import plotly.express as px

                        df = px.data.medals_wide(indexed=True)
                        fig = px.imshow(retscomp.corr(),color_continuous_scale='YlOrRd',labels=dict(x="Correlação"))
                        #fig.update_layout(autosize=False,width=1200,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')
                        fig.update_xaxes(side="top")

                        st.plotly_chart(fig,use_container_width=True)   
                except:
                        exit

        # ------------------------------ GRÁFICOS DE mapa de risco-------------------------- 

                map = returns.reset_index()
                layout = go.Layout(title=f"Mapa de Risco x Retorno",xaxis=dict(title="Retorno esperado"), yaxis=dict(title="Risco"))
                fig = go.Figure(layout = layout)
                for i in range(len(map['symbol'].unique())):
                        fig.add_trace(go.Scatter(x=[map.loc[map['symbol']==map['symbol'].unique()[i]]['close'].mean() * 100], y=[map.loc[map['symbol']==map['symbol'].unique()[i]]['close'].std() * 100],name=map['symbol'].unique()[i],marker=dict(size=30)))
                #fig.add_trace(go.Scatter(x=[map['close'].mean()], y=[map['close'].std()],text=map['symbol'].unique()))
                fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Red')#, range=[-0.005, 0.01])
                fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Red')#, range=[-0.01, 0.1])
                fig.update_traces(textposition='top center')
                #fig.update_layout(autosize=False,width=800,height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

                fig.update_layout( height=500, showlegend=False, paper_bgcolor='rgba(255,255,255,0.9)', plot_bgcolor='rgba(255,255,255,0.9)') 
                fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor = 'rgb(240,238,238)')

                st.plotly_chart(fig,use_container_width=True)   

