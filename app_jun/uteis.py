
#import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def rank( X, y):
    """
    Gets a rank of relation of features to target
    The output is a list of combination of features order by relevance by 'RF rank'
    
    :param X: Train data without target
    :param y: Target
    
    """

    X = X
    y = y
    # estimators
    rank = pd.DataFrame({'features': X.columns})


    rfr = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
    rfr.fit(X, y)
    rank['RFR'] = (rfr.feature_importances_ * 100)

    #print(rank.sort_values('RFR', ascending=False))

    # opções de listas de features selecionadas para cada estimador
    lista_comb_feat_RFR = []
    #for x in range(2, 11):
    for x in range(2, len(rank)):
        lista_comb_feat_RFR.append(rank.sort_values('RFR', ascending=False).head(x)['features'].tolist())

    return lista_comb_feat_RFR , rank.sort_values('RFR', ascending=False)

#UTILIZA LISTA COM NOMES ÚNICOS DAS AÇÕES, DATAFRAME DA B3, DADOS HISTÓRICOS E A TABELA DE INPUT COM UMA LINHA PARA CADA AÇÃO
def inputer_train(lista, df, data, df_filled):

    for i in range(len(lista)):
        if df.loc[(df['Código de Negociação'].str.contains(lista[i]))].sort_values('Tipo de Movimentação')[-1:]['Tipo de Movimentação'].item() == 'Venda':
            
            #Construção das variáveis
            preco_medio_compra = df.loc[(df['Código de Negociação'].str.contains(lista[i])) & (df['Tipo de Movimentação'] == 'Compra' )]['Preço'].mean()
            quantidade_comprada = df.loc[(df['Código de Negociação'].str.contains(lista[i])) & (df['Tipo de Movimentação'] == 'Compra' )]['Quantidade'].sum()
            preco_medio_vendido = df.loc[(df['Código de Negociação'].str.contains(lista[i])) & (df['Tipo de Movimentação'] == 'Venda' )]['Preço'].mean()
            quantidade_vendida = df.loc[(df['Código de Negociação'].str.contains(lista[i])) & (df['Tipo de Movimentação'] == 'Venda' )]['Quantidade'].sum()
            
            data_compra_1 = df.loc[(df['Código de Negociação'].str.contains(lista[i]))].sort_values('Data do Negócio')[-1:]['Data do Negócio'].item()
            Ganho_total = (preco_medio_vendido - preco_medio_compra) * quantidade_vendida
            rendimento_total = round(((preco_medio_vendido - preco_medio_compra) / preco_medio_vendido) * 100,2)
            
            #dados históricos ticker
            dados_acao = pd.DataFrame(data.loc[ : , (['Open','High','Low','Close','Adj Close','Volume'],lista[i]+".SA")])
            #dados_acao_filtrado = dados_acao.loc[dados_acao.index <= data_compra_1]
            dados_acao_filtrado = dados_acao.loc[dados_acao.index <= pd.to_datetime(data_compra_1)]               
            
            #RENDIMENTO ULTIMOS X DIAS (ONTEM X DIA COMPARADO)
            
            #valor da ultima cotação
            cotacao_last = dados_acao_filtrado['Close'][-1:][lista[i]+".SA"][0]
            #valor cotação x dias atras
            try:
                cotacao_7 = dados_acao_filtrado['Close'][-7:-6][lista[i]+".SA"][0]
                cotacao_14 = dados_acao_filtrado['Close'][-14:-13][lista[i]+".SA"][0]
                cotacao30 = dados_acao_filtrado['Close'][-30:-29][lista[i]+".SA"][0]
                cotacao_60 = dados_acao_filtrado['Close'][-60:-59][lista[i]+".SA"][0]
                cotacao_90 = dados_acao_filtrado['Close'][-90:-89][lista[i]+".SA"][0]
                #% da queda ou aumento ultimos x dias
                crescimento_7 = round(((cotacao_last - cotacao_7) / cotacao_7) * 100,2)
                crescimento_14 = round(((cotacao_last - cotacao_14) / cotacao_14) * 100,2)
                crescimento_30 = round(((cotacao_last - cotacao30) / cotacao30) * 100,2)
                crescimento_60 = round(((cotacao_last - cotacao_60) / cotacao_60) * 100,2)
                crescimento_90 = round(((cotacao_last - cotacao_90) / cotacao_90) * 100,2)
            
            except:
                exit

            #CRESCIMENTO VOLUME ULTIMOS X DIAS (ONTEM X DIA COMPARADO)
            
            #volume do dia anterior a compra
            volume_last = dados_acao_filtrado['Volume'][-1:][lista[i]+".SA"][0]
            #valor cotação x dias atras
            try:
                volume_7 = dados_acao_filtrado['Volume'][-7:-6][lista[i]+".SA"][0]
                volume_14 = dados_acao_filtrado['Volume'][-14:-13][lista[i]+".SA"][0]
                volume30 = dados_acao_filtrado['Volume'][-30:-29][lista[i]+".SA"][0]
                volume_60 = dados_acao_filtrado['Volume'][-60:-59][lista[i]+".SA"][0]
                volume_90 = dados_acao_filtrado['Volume'][-90:-89][lista[i]+".SA"][0]
                #% da queda ou aumento ultimos x dias
                crescimento_vol_7 = round(((volume_last - volume_7) / volume_7) * 100,2)
                crescimento_vol_14 = round(((volume_last - volume_14) / volume_14) * 100,2)
                crescimento_vol_30 = round(((volume_last - volume30) / volume30) * 100,2)
                crescimento_vol_60 = round(((volume_last - volume_60) / volume_60) * 100,2)
                crescimento_vol_90 = round(((volume_last - volume_90) / volume_90) * 100,2)
            
            except:
                exit
                
            #RSI
            try:
                delta = dados_acao_filtrado['Close'][-90:].diff()
                up, down = delta.copy(), delta.copy()
                up[up < 0] = 0
                down[down > 0] = 0
                period = 14
                rUp = up.ewm(com=period - 1,  adjust=False).mean()
                rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
                delta['RSI'] = 100 - 100 / (1 + rUp / rDown).fillna(0)
                
                rsi_0 = delta['RSI'][-1:][0]
                rsi_7 = delta['RSI'][-7:-6][0]
                rsi_14 = delta['RSI'][-14:-13][0]
                rsi_30 = delta['RSI'][-30:-29][0]
                rsi_60 = delta['RSI'][-60:-59][0]

                #% da queda ou aumento ultimos x dias
                cresc_rsi_7 = round(((rsi_0 - rsi_7) / rsi_7) * 100,2)
                cresc_rsi_14 = round(((rsi_0 - rsi_14) / rsi_14) * 100,2)
                cresc_rsi_30 = round(((rsi_0 - rsi_30) / rsi_30) * 100,2)
                cresc_rsi_60 = round(((rsi_0 - rsi_60) / rsi_60) * 100,2)

            except:
                exit
                
            #BOLINGER
            
            try:
                bolinger = dados_acao_filtrado.copy()
                bolinger['MA20'] = dados_acao_filtrado['Close'].rolling(20).mean()
                bolinger['20 Day STD'] = bolinger['Close'].rolling(window=20).std()
                bolinger['Upper Band'] = bolinger['MA20'] + (bolinger['20 Day STD'] * 2)
                bolinger['Lower Band'] = bolinger['MA20'] - (bolinger['20 Day STD'] * 2)

                boolinger_up_0 = bolinger['Upper Band'][-1:][0]
                boolinger_down_0 = bolinger['Lower Band'][-1:][0]
                boolinger_up_7 = bolinger['Upper Band'][-7:-6][0]
                boolinger_down_7 = bolinger['Lower Band'][-7:-6][0]

                delta_bolinger_0 = round((boolinger_up_0 - boolinger_down_0) / boolinger_down_0 * 100,2)
                cresc_bolinger_up_7 = round((boolinger_up_0 - boolinger_up_7) / boolinger_up_7 * 100,2)
                cresc_bolinger_down_7 = round((boolinger_down_0 - boolinger_down_7) / boolinger_down_7 * 100,2)
                
            except:
                exit
                
            
            #MÉDIAS MOVEIS
            try:
                time = dados_acao_filtrado.copy()
                rolling_9  = time['Close'].rolling(window=9)
                rolling_mean_9 = rolling_9.mean().round(1)

                rolling_20  = time['Close'].rolling(window=20)
                rolling_mean_20 = rolling_20.mean().round(1)

                rolling_72  = time['Close'].rolling(window=72)
                rolling_mean_72 = rolling_72.mean().round(1)
                time['MM9'] = rolling_mean_9.fillna(0)
                time['MM20'] = rolling_mean_20.fillna(0)
                time['MM72'] = rolling_mean_72.fillna(0)
                time['cruzamento'] =  time['MM9'] - time['MM72']
                buy = time.tail(50).loc[(time.tail(50)['cruzamento']==0)]

                if buy.empty == False:
                    cruzou_mm = 1
                else:
                    cruzou_mm = 0         

                if time['MM72'].iloc[-1] < time['MM9'].iloc[-1]:
                    direcao_cruzada_cima = 1
                else:
                    direcao_cruzada_cima = 0
                    

                mm9_0 = time['MM9'][-1:][0]
                mm9_7 = time['MM9'][-7:-6][0]
                mm9_14 = time['MM9'][-14:-13][0]
                mm9_30 = time['MM9'][-30:-29][0]
                mm9_60 = time['MM9'][-60:-59][0]
                
                mm20_0 = time['MM20'][-1:][0]
                mm20_7 = time['MM20'][-7:-6][0]
                mm20_14 = time['MM20'][-14:-13][0]
                mm20_30 = time['MM20'][-30:-29][0]
                mm20_60 = time['MM20'][-60:-59][0]
    
                mm72_0 = time['MM72'][-1:][0]
                mm72_7 = time['MM72'][-7:-6][0]
                mm72_14 = time['MM72'][-14:-13][0]
                mm72_30 = time['MM72'][-30:-29][0]
                mm72_60 = time['MM72'][-60:-59][0]
                
                #% da queda ou aumento ultimos x dias
                cresc_mm9_7 = round(((mm9_0 - mm9_7) / mm9_7) * 100,2)
                cresc_mm9_14 = round(((mm9_0 - mm9_14) / mm9_14) * 100,2)
                cresc_mm9_30 = round(((mm9_0 - mm9_30) / mm9_30) * 100,2)
                cresc_mm9_60 = round(((mm9_0 - mm9_60) / mm9_60) * 100,2)
                
                #% da queda ou aumento ultimos x dias
                cresc_mm20_7 = round(((mm20_0 - mm20_7) / mm20_7) * 100,2)
                cresc_mm20_14 = round(((mm20_0 - mm20_14) / mm20_14) * 100,2)
                cresc_mm20_30 = round(((mm20_0 - mm20_30) / mm20_30) * 100,2)
                cresc_mm20_60 = round(((mm20_0 - mm20_60) / mm20_60) * 100,2)
                
                #% da queda ou aumento ultimos x dias
                cresc_mm72_7 = round(((mm72_0 - mm72_7) / mm72_7) * 100,2)
                cresc_mm72_14 = round(((mm72_0 - mm72_14) / mm72_14) * 100,2)
                cresc_mm72_30 = round(((mm72_0 - mm72_30) / mm72_30) * 100,2)
                cresc_mm72_60 = round(((mm72_0 - mm72_60) / mm72_60) * 100,2)
                
            except:
                exit
                
            #try:
            #    pfizer = yf.Ticker(lista[i])
            #    info = pfizer.info 

            #    setor = info['sector']
            #    atividade = info['industry']
                
            #except:
            #    exit

            try:
            #Atribuições
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'data_compra_1'] = data_compra_1
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Preço_médio_comprado'] = preco_medio_compra
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Preço_médio_vendido'] = preco_medio_vendido
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Ganho_total'] = Ganho_total

                #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Setor'] = setor
                #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Atividade'] = atividade

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_total_%'] = rendimento_total
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_7_dias'] = crescimento_7
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_14_dias'] = crescimento_14
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_30_dias'] = crescimento_30
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_60_dias'] = crescimento_60
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_90_dias'] = crescimento_90

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_7_dias'] = crescimento_vol_7
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_14_dias'] = crescimento_vol_14
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_30_dias'] = crescimento_vol_30
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_60_dias'] = crescimento_vol_60
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_90_dias'] = crescimento_vol_90

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'rsi'] = rsi_0
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_7_dias'] = cresc_rsi_7
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_14_dias'] = cresc_rsi_14
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_30_dias'] = cresc_rsi_30
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_60_dias'] = cresc_rsi_60

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'delta_bolinger_0'] = delta_bolinger_0
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_bolinger_up_7'] = cresc_bolinger_up_7
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_bolinger_down_7'] = cresc_bolinger_down_7

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cruzou_mm'] = cruzou_mm
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'direcao_cruzada_mm_cima'] = direcao_cruzada_cima

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_7_dias'] = cresc_mm9_7
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_14_dias'] = cresc_mm9_14
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_30_dias'] = cresc_mm9_30
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_60_dias'] = cresc_mm9_60

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_7_dias'] = cresc_mm20_7
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_14_dias'] = cresc_mm20_14
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_30_dias'] = cresc_mm20_30
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_60_dias'] = cresc_mm20_60        

                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_7_dias'] = cresc_mm72_7
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_14_dias'] = cresc_mm72_14
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_30_dias'] = cresc_mm72_30
                df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_60_dias'] = cresc_mm72_60
            
            except:
                exit

def inputer_predict(data, df_filled):

    lista = list(df_filled['name'])
    for i in range(len(df_filled)):
        dados_acao = pd.DataFrame(data.loc[ : , (['Open','High','Low','Close','Adj Close','Volume'],lista[i]+".SA")])
        dados_acao_filtrado = dados_acao.copy()
        #RENDIMENTO ULTIMOS X DIAS (ONTEM X DIA COMPARADO)
        #valor da ultima cotação
        cotacao_last = dados_acao_filtrado['Close'][-1:][lista[i]+".SA"][0]
        #valor cotação x dias atras
        try:
            cotacao_7 = dados_acao_filtrado['Close'][-7:-6][lista[i]+".SA"][0]
            cotacao_14 = dados_acao_filtrado['Close'][-14:-13][lista[i]+".SA"][0]
            cotacao30 = dados_acao_filtrado['Close'][-30:-29][lista[i]+".SA"][0]
            cotacao_60 = dados_acao_filtrado['Close'][-60:-59][lista[i]+".SA"][0]
            cotacao_90 = dados_acao_filtrado['Close'][-90:-89][lista[i]+".SA"][0]
            #% da queda ou aumento ultimos x dias
            crescimento_7 = round(((cotacao_last - cotacao_7) / cotacao_7) * 100,2)
            crescimento_14 = round(((cotacao_last - cotacao_14) / cotacao_14) * 100,2)
            crescimento_30 = round(((cotacao_last - cotacao30) / cotacao30) * 100,2)
            crescimento_60 = round(((cotacao_last - cotacao_60) / cotacao_60) * 100,2)
            crescimento_90 = round(((cotacao_last - cotacao_90) / cotacao_90) * 100,2)
        except:
            exit
        #CRESCIMENTO VOLUME ULTIMOS X DIAS (ONTEM X DIA COMPARADO)
        #volume do dia anterior a compra
        volume_last = dados_acao_filtrado['Volume'][-1:][lista[i]+".SA"][0]
        #valor cotação x dias atras
        try:
            volume_7 = dados_acao_filtrado['Volume'][-7:-6][lista[i]+".SA"][0]
            volume_14 = dados_acao_filtrado['Volume'][-14:-13][lista[i]+".SA"][0]
            volume30 = dados_acao_filtrado['Volume'][-30:-29][lista[i]+".SA"][0]
            volume_60 = dados_acao_filtrado['Volume'][-60:-59][lista[i]+".SA"][0]
            volume_90 = dados_acao_filtrado['Volume'][-90:-89][lista[i]+".SA"][0]
            #% da queda ou aumento ultimos x dias
            crescimento_vol_7 = round(((volume_last - volume_7) / volume_7) * 100,2)
            crescimento_vol_14 = round(((volume_last - volume_14) / volume_14) * 100,2)
            crescimento_vol_30 = round(((volume_last - volume30) / volume30) * 100,2)
            crescimento_vol_60 = round(((volume_last - volume_60) / volume_60) * 100,2)
            crescimento_vol_90 = round(((volume_last - volume_90) / volume_90) * 100,2)
        except:
            exit
        #RSI
        try:
            delta = dados_acao_filtrado['Close'][-90:].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            period = 14
            rUp = up.ewm(com=period - 1,  adjust=False).mean()
            rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
            delta['RSI'] = 100 - 100 / (1 + rUp / rDown).fillna(0)

            rsi_0 = delta['RSI'][-1:][0]
            rsi_7 = delta['RSI'][-7:-6][0]
            rsi_14 = delta['RSI'][-14:-13][0]
            rsi_30 = delta['RSI'][-30:-29][0]
            rsi_60 = delta['RSI'][-60:-59][0]

            #% da queda ou aumento ultimos x dias
            cresc_rsi_7 = round(((rsi_0 - rsi_7) / rsi_7) * 100,2)
            cresc_rsi_14 = round(((rsi_0 - rsi_14) / rsi_14) * 100,2)
            cresc_rsi_30 = round(((rsi_0 - rsi_30) / rsi_30) * 100,2)
            cresc_rsi_60 = round(((rsi_0 - rsi_60) / rsi_60) * 100,2)
        except:
            exit
        #BOLINGER
        try:
            bolinger = dados_acao_filtrado.copy()
            bolinger['MA20'] = dados_acao_filtrado['Close'].rolling(20).mean()
            bolinger['20 Day STD'] = bolinger['Close'].rolling(window=20).std()
            bolinger['Upper Band'] = bolinger['MA20'] + (bolinger['20 Day STD'] * 2)
            bolinger['Lower Band'] = bolinger['MA20'] - (bolinger['20 Day STD'] * 2)

            boolinger_up_0 = bolinger['Upper Band'][-1:][0]
            boolinger_down_0 = bolinger['Lower Band'][-1:][0]
            boolinger_up_7 = bolinger['Upper Band'][-7:-6][0]
            boolinger_down_7 = bolinger['Lower Band'][-7:-6][0]

            delta_bolinger_0 = round((boolinger_up_0 - boolinger_down_0) / boolinger_down_0 * 100,2)
            cresc_bolinger_up_7 = round((boolinger_up_0 - boolinger_up_7) / boolinger_up_7 * 100,2)
            cresc_bolinger_down_7 = round((boolinger_down_0 - boolinger_down_7) / boolinger_down_7 * 100,2)
        except:
            exit
        #MÉDIAS MOVEIS
        try:
            time = dados_acao_filtrado.copy()
            rolling_9  = time['Close'].rolling(window=9)
            rolling_mean_9 = rolling_9.mean().round(1)

            rolling_20  = time['Close'].rolling(window=20)
            rolling_mean_20 = rolling_20.mean().round(1)

            rolling_72  = time['Close'].rolling(window=72)
            rolling_mean_72 = rolling_72.mean().round(1)
            time['MM9'] = rolling_mean_9.fillna(0)
            time['MM20'] = rolling_mean_20.fillna(0)
            time['MM72'] = rolling_mean_72.fillna(0)
            time['cruzamento'] =  time['MM9'] - time['MM72']
            buy = time.tail(50).loc[(time.tail(50)['cruzamento']==0)]

            if buy.empty == False:
                cruzou_mm = 1
            else:
                cruzou_mm = 0         

            if time['MM72'].iloc[-1] < time['MM9'].iloc[-1]:
                direcao_cruzada_cima = 1
            else:
                direcao_cruzada_cima = 0
                
            mm9_0 = time['MM9'][-1:][0]
            mm9_7 = time['MM9'][-7:-6][0]
            mm9_14 = time['MM9'][-14:-13][0]
            mm9_30 = time['MM9'][-30:-29][0]
            mm9_60 = time['MM9'][-60:-59][0]

            mm20_0 = time['MM20'][-1:][0]
            mm20_7 = time['MM20'][-7:-6][0]
            mm20_14 = time['MM20'][-14:-13][0]
            mm20_30 = time['MM20'][-30:-29][0]
            mm20_60 = time['MM20'][-60:-59][0]

            mm72_0 = time['MM72'][-1:][0]
            mm72_7 = time['MM72'][-7:-6][0]
            mm72_14 = time['MM72'][-14:-13][0]
            mm72_30 = time['MM72'][-30:-29][0]
            mm72_60 = time['MM72'][-60:-59][0]

            #% da queda ou aumento ultimos x dias
            cresc_mm9_7 = round(((mm9_0 - mm9_7) / mm9_7) * 100,2)
            cresc_mm9_14 = round(((mm9_0 - mm9_14) / mm9_14) * 100,2)
            cresc_mm9_30 = round(((mm9_0 - mm9_30) / mm9_30) * 100,2)
            cresc_mm9_60 = round(((mm9_0 - mm9_60) / mm9_60) * 100,2)

            #% da queda ou aumento ultimos x dias
            cresc_mm20_7 = round(((mm20_0 - mm20_7) / mm20_7) * 100,2)
            cresc_mm20_14 = round(((mm20_0 - mm20_14) / mm20_14) * 100,2)
            cresc_mm20_30 = round(((mm20_0 - mm20_30) / mm20_30) * 100,2)
            cresc_mm20_60 = round(((mm20_0 - mm20_60) / mm20_60) * 100,2)

            #% da queda ou aumento ultimos x dias
            cresc_mm72_7 = round(((mm72_0 - mm72_7) / mm72_7) * 100,2)
            cresc_mm72_14 = round(((mm72_0 - mm72_14) / mm72_14) * 100,2)
            cresc_mm72_30 = round(((mm72_0 - mm72_30) / mm72_30) * 100,2)
            cresc_mm72_60 = round(((mm72_0 - mm72_60) / mm72_60) * 100,2)

        except:
            exit

        try:

        #Atribuições
            #df_filled.loc[df_filled['name'].str.contains(lista[i]),'data_compra_1'] = data_compra_1
            #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Preço_médio_comprado'] = preco_medio_compra
            #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Preço_médio_vendido'] = preco_medio_vendido
            #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Ganho_total'] = Ganho_total

            #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Setor'] = setor
            #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Atividade'] = atividade

            #df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_total_%'] = rendimento_total
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_7_dias'] = crescimento_7
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_14_dias'] = crescimento_14
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_30_dias'] = crescimento_30
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_60_dias'] = crescimento_60
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'Rendimento_ultimos_90_dias'] = crescimento_90

            df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_7_dias'] = crescimento_vol_7
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_14_dias'] = crescimento_vol_14
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_30_dias'] = crescimento_vol_30
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_60_dias'] = crescimento_vol_60
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'crescimento_vol_ultimos_90_dias'] = crescimento_vol_90

            df_filled.loc[df_filled['name'].str.contains(lista[i]),'rsi'] = rsi_0
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_7_dias'] = cresc_rsi_7
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_14_dias'] = cresc_rsi_14
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_30_dias'] = cresc_rsi_30
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_rsi_ultimos_60_dias'] = cresc_rsi_60

            df_filled.loc[df_filled['name'].str.contains(lista[i]),'delta_bolinger_0'] = delta_bolinger_0
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_bolinger_up_7'] = cresc_bolinger_up_7
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_bolinger_down_7'] = cresc_bolinger_down_7

            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cruzou_mm'] = cruzou_mm
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'direcao_cruzada_mm_cima'] = direcao_cruzada_cima

            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_7_dias'] = cresc_mm9_7
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_14_dias'] = cresc_mm9_14
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_30_dias'] = cresc_mm9_30
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm9_ultimos_60_dias'] = cresc_mm9_60

            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_7_dias'] = cresc_mm20_7
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_14_dias'] = cresc_mm20_14
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_30_dias'] = cresc_mm20_30
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm20_ultimos_60_dias'] = cresc_mm20_60        

            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_7_dias'] = cresc_mm72_7
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_14_dias'] = cresc_mm72_14
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_30_dias'] = cresc_mm72_30
            df_filled.loc[df_filled['name'].str.contains(lista[i]),'cresc_mm72_ultimos_60_dias'] = cresc_mm72_60

            
        except:
            exit
        
    return df_filled

