import matplotlib
matplotlib.use('Agg')

import streamlit as st


import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import datetime as dt 
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



import uteis as uteis
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


def analise_carteira():
    top_ativos = pd.read_excel('data/top_200.xlsx')#, index_col=0) 

    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        #st.image('https://media.giphy.com/media/3ShFD4IvX96027JjhH/giphy.gif', width=400)
        st.write("")
    with col3:
        st.write("")

    #st.image('https://media.giphy.com/media/3ShFD4IvX96027JjhH/giphy.gif', width=400)    
    #image = Image.open('imagens/logo.jpg')
    #st.image(image, use_column_width=True)                       
    st.title('Análise de carteira e previsão de lucro')
    st.subheader('Receba insights sobre suas operações realizadas no passado e preveja se sua próxima operação no futuro será lucrativa, ou não!')
    st.write('Usando os dados do seu extrato histórico fornecido pelo site da B3 iremos treinar um algorítimo de inteligência artificial que será capaz de analisar suas operações passadas, mostrar padrões que te levaram ao lucro ou prejuízo, além de prever a probabilidade de lucro de uma ação caso ela seja comprada hoje por você')

    menu = ["Escolha uma opção","Testar com nossos dados!","Usar os dados de minhas operações"]
    choice = st.selectbox("Menu",menu)
    
    if choice == "Usar os dados de minhas operações":
        st.subheader("Login Section")

        username = st.text_input("User Name")
        password = st.text_input("Password",type='password')
        if password == '12345':
            st.subheader('Faça upload aqui do seu extrato da B3')
            with st.expander("Passo a passo de como acessar os dados no site da B3"):
                st.write('Acessar o site www.investidorb3.com.br')
                st.write('Aba Extratos > Negociação > Aplicar filtro trazendo dados do último ano > baixar extrato em formato excel')
                image = Image.open('images/b3.png')
                st.image(image, use_column_width=True)    

            st.subheader('Faça upload aqui do seu extrato da B3')
            file  = st.file_uploader('Entre com seu extrato (.xlsx)', type = 'xlsx')    
            if file:
                df = pd.read_excel(file)

                #st.dataframe(df)
                #st.table(df)
                #st.write('Lucro Total até hoje: R$',round(df['Preço Médio (Venda)'].sum(),2))  
                lista = []
                retirar = []
                for i in range(len(df['Código de Negociação'])):
                    if len(df.iloc[i]['Código de Negociação']) == 5:
                        lista.append(df.iloc[i]['Código de Negociação'])
                    
                    elif df.iloc[i]['Código de Negociação'][-1] == '1':
                        retirar.append(df.iloc[i]['Código de Negociação'][-1])
                    else:
                        lista.append(df.iloc[i]['Código de Negociação'][:-1])

                lista = pd.DataFrame(lista)[0].unique()
                lista_input = []
                for i in range(len(lista)):
                    
                    lista_input.append(str(lista[i] + '.SA'))

                date_year_ago = dt.datetime.today() - dt.timedelta(days=565)
                date_year_ago = date_year_ago.strftime(format='20%y-%m-%d')
                data = yf.download(lista_input,start=date_year_ago)

                df_filled = pd.DataFrame(columns = ['name'])
                df_filled['name'] = lista_input

                # lógica para input de dados calculados 




                uteis.inputer_train(lista, df, data, df_filled)



                
                df_input = df_filled.fillna(0).replace(np.inf, 0)

                st.subheader('Avaliação de carteira:')
                st.write('Lucro Total do período avaliado: R$',round(df_input['Ganho_total'].sum(),2))
                #st.write('Rendimento Total do período avaliado: %',round(df_input['Rendimento_total_%'].sum(),2))

                df_input = df_input.loc[df_input['data_compra_1'] != 0]
                df_input['data_compra_1'] = pd.to_datetime(df_input['data_compra_1']).copy()
                df_ordered = df_input.sort_values('data_compra_1')
                #ordenando e criando campo mes ano
                df_ordered['mes/ano'] =df_ordered['data_compra_1'].astype(str).str[:-3]
                df_grouped = df_ordered.groupby('mes/ano').agg({'Rendimento_total_%':'mean','Ganho_total':'sum'})
                df_grouped = df_grouped.reset_index()
                
                #from plotly.subplots import make_subplots
                #fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scatter"}, {"type": "bar"}]], subplot_titles=("Rendimento mensal %","Lucro total mensal R$") )
                #fig.add_trace(go.Scatter(x =df_grouped['mes/ano'],  y=df_grouped['Rendimento_total_%']), row=1, col=1)
                #fig.add_trace(go.Bar(x =df_grouped['mes/ano'],  y=df_grouped['Ganho_total']), row=1, col=2)
                #fig.update_layout(height=800, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                #st.plotly_chart(fig)

                layout = go.Layout(title="Rendimento mensal %",xaxis=dict(title="mês/ano"), yaxis=dict(title="Rendimento total %"))
                fig = go.Figure(layout = layout)
                fig.add_trace(go.Scatter(x =df_grouped['mes/ano'],  y=df_grouped['Rendimento_total_%']))
                fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

                st.plotly_chart(fig)

                layout = go.Layout(title="Lucro total mensal R$",xaxis=dict(title="mês/ano"), yaxis=dict(title="Ganho total R$"))
                fig = go.Figure(layout = layout)
                fig.add_trace(go.Bar(x =df_grouped['mes/ano'],  y=df_grouped['Ganho_total']))
                fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

                st.plotly_chart(fig)

                #MODELAGEM
                
                df_ordered['lucro'] = 0
                df_ordered.loc[df_ordered['Ganho_total'] > 0 , 'lucro'] = 1
                df_ordered = df_ordered.fillna(0).replace(-np.inf, 0)

                X = df_ordered.drop(['name', 'data_compra_1','mes/ano','Ganho_total','Rendimento_total_%','lucro','Preço_médio_comprado','Preço_médio_vendido'],axis=1)
                y = df_ordered['lucro']

                # divisão entre treino e teste 70/30
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

                # Random Forest Regressor MVP
                regr = RandomForestRegressor(random_state=42)
                regr.fit(X_train,y_train)

                predictions = regr.predict(X_test)

                comparar = pd.DataFrame(y_test)
                comparar['previsto'] = predictions
                comparar['dif'] = comparar['previsto'] - comparar['lucro']

                erros = len(comparar.loc[comparar['dif'] > 0.5]) + len(comparar.loc[comparar['dif'] < -0.5])
                total = len(comparar)
                precision_model = round(1 - (erros / total),2)

                st.write('O modelo criado com os seus dados tem uma precisão de acerto de: ',precision_model * 100 ,'%')
                st.write('Caso a precisão seja baixa ( < 65% )  é necessário mais dados para melhorar a performance do modelo, neste caso utilize nosso modelo pré treinado na opção " Testar com nossos dados" ou incremente seus dados com operações fictícias')

                #trazendo features + importantes

                features, rank = uteis.rank(X, y)

                st.subheader('Estas variáveis são as que mais impactam nas decisões da inteligência artificial')

                st.table(rank['features'].head(10).reset_index(drop=True))


                #fazendo previsão em toda a bolsa

                #lista = scraping.get_data()
                #todos = pd.DataFrame(flatten(lista).keys()).transpose()
                #todos.columns = todos.iloc[0]
                #for i in range(len(lista)): 
                #    todos = pd.concat([todos,pd.DataFrame(lista[i]).transpose()])

                #todos = todos.iloc[1:]
                #todos['name'] = (todos.index + '.SA' )
                lista = top_ativos.copy()
                top_ativos['name'] =top_ativos['name'] + '.SA' 
                todos = top_ativos.copy()

                date_year_ago = dt.datetime.today() - dt.timedelta(days=300)
                date_year_ago = date_year_ago.strftime(format='20%y-%m-%d')


                data = yf.download(list(todos['name']),start=date_year_ago)

                df_filled = pd.DataFrame(columns = ['name'])
                df_filled['name'] = lista['name']

                df_filled['ativo'] = df_filled['name'].copy()
                df_filled = df_filled.set_index('ativo')

                uteis.inputer_predict(data, df_filled)


                df_filled = uteis.inputer_predict(data, df_filled)
                # retirar nulos e infinitos positivos
                df_input = df_filled.fillna(0).replace(np.inf, 0)
                #df_ordered = df_input.fillna(0).replace(-np.inf, 0)
                input_predict = df_input[list(X.columns)]
                # retirar nulos e infinitos negativos
                input_predict = input_predict.fillna(0).replace(-np.inf, 0)
                predictions = regr.predict(input_predict)
                input_predict['probabilidade de lucro'] = predictions.round(2) * 100

                st.subheader('Previsão de probabilidade das principais ações da bolsa')
                st.text('Essa previsão é feita com base nas tendências de sucesso captadas pelas suas operações')

                st.table(input_predict['probabilidade de lucro'].sort_values(ascending=False).round(2))







        else:
            st.warning("Incorrect Username/Password")
    

    if choice == "Testar com nossos dados!":
        st.subheader('As previsões feitas aqui utilizam dados de movimentação de uma carteira fictícia para exemplificar o funcionamento da inteligência artificial')

            







        