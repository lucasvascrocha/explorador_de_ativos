#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import pandas as pd
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


from yahooquery import Ticker
from fbprophet import Prophet
import yfinance as yf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,LSTM
#import stocker
 
import datetime as dt 
dia = dt.datetime.today().strftime(format='20%y-%m-%d')



import re
import urllib.request
import urllib.parse
import http.cookiejar
import time
import lxml

from lxml.html import fragment_fromstring
from collections import OrderedDict
import json
import ast
import datetime
import os
from pymongo import MongoClient

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# ------------------------------ SCRAPPING -------------------------------


def get_data(*args, **kwargs):
    class AppURLopener(urllib.request.FancyURLopener):
      version = "Mozilla/5.0"

    opener = AppURLopener()
    response = opener.open('http://httpbin.org/user-agent')

    url = 'http://www.fundamentus.com.br/resultado.php'
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'),
                        ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')                                                             
                        ]

    #opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201'),
    #                     ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')]

    # Aqui estão os parâmetros de busca das ações
    # Estão em branco para que retorne todas as disponíveis
    data = {'pl_min':'','pl_max':'','pvp_min':'','pvp_max' :'','psr_min':'','psr_max':'','divy_min':'','divy_max':'',            'pativos_min':'','pativos_max':'','pcapgiro_min':'','pcapgiro_max':'','pebit_min':'','pebit_max':'', 'fgrah_min':'',
            'fgrah_max':'', 'firma_ebit_min':'', 'firma_ebit_max':'','margemebit_min':'','margemebit_max':'',            'margemliq_min':'','margemliq_max':'', 'liqcorr_min':'','liqcorr_max':'','roic_min':'','roic_max':'','roe_min':'',            'roe_max':'','liq_min':'','liq_max':'','patrim_min':'','patrim_max':'','divbruta_min':'','divbruta_max':'',         'tx_cresc_rec_min':'','tx_cresc_rec_max':'','setor':'','negociada':'ON','ordem':'1','x':'28','y':'16'}

    with opener.open(url, urllib.parse.urlencode(data).encode('UTF-8')) as link:
        content = link.read().decode('ISO-8859-1')

    pattern = re.compile('<table id="resultado".*</table>', re.DOTALL)
    reg = re.findall(pattern, content)[0]
    page = fragment_fromstring(reg)
    lista = OrderedDict()


    stocks = page.xpath('tbody')[0].findall("tr")

    todos = []
    for i in range(0, len(stocks)):
        lista[i] = {
            stocks[i].getchildren()[0][0].getchildren()[0].text: {
                'cotacao': stocks[i].getchildren()[1].text,
               'P/L': stocks[i].getchildren()[2].text,
               'P/VP': stocks[i].getchildren()[3].text,
               'PSR': stocks[i].getchildren()[4].text,
               'DY': stocks[i].getchildren()[5].text,
               'P/Ativo': stocks[i].getchildren()[6].text,
               'P/Cap.Giro': stocks[i].getchildren()[7].text,
               'P/EBIT': stocks[i].getchildren()[8].text,
               'P/Ativ.Circ.Liq.': stocks[i].getchildren()[9].text,
               'EV/EBIT': stocks[i].getchildren()[10].text,
               'EBITDA': stocks[i].getchildren()[11].text,
               'Mrg. Ebit': stocks[i].getchildren()[12].text,
               'Mrg.Liq.': stocks[i].getchildren()[13].text,
               'Liq.Corr.': stocks[i].getchildren()[14].text,
               'ROIC': stocks[i].getchildren()[15].text,
               'ROE': stocks[i].getchildren()[16].text,
               'Liq.2m.': stocks[i].getchildren()[17].text,
               'Pat.Liq': stocks[i].getchildren()[18].text,
               'Div.Brut/Pat.': stocks[i].getchildren()[19].text,
               'Cresc.5a': stocks[i].getchildren()[20].text
               }
            }

    return lista

def get_specific_data(stock):
    class AppURLopener(urllib.request.FancyURLopener):
      version = "Mozilla/5.0"

    opener = AppURLopener()
    response = opener.open('http://httpbin.org/user-agent')

    url = "http://www.fundamentus.com.br/detalhes.php?papel=" + stock
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'),
                        ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')                                                             
                        ]
    
    
    
    
    
    #opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201'),
    #                     ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')]
    
    # Get data from site
    link = opener.open(url, urllib.parse.urlencode({}).encode('UTF-8'))
    content = link.read().decode('ISO-8859-1')

    # Get all table instances
    pattern = re.compile('<table class="w728">.*</table>', re.DOTALL)
    reg = re.findall(pattern, content)[0]
    reg = "<div>" + reg + "</div>"
    page = fragment_fromstring(reg)
    all_data = {}

    # There is 5 tables with tr, I will get all trs
    all_trs = []
    all_tables = page.xpath("table")

    for i in range(0, len(all_tables)):
        all_trs = all_trs + all_tables[i].findall("tr")

    # Run through all the trs and get the label and the
    # data for each line
    for tr_index in range(0, len(all_trs)):
        tr = all_trs[tr_index]
        # Get into td
        all_tds = tr.getchildren()
        for td_index in range(0, len(all_tds)):
            td = all_tds[td_index]

            label = ""
            data = ""

            # The page has tds with contents and some 
            # other with not
            if (td.get("class").find("label") != -1):
                # We have a label
                for span in td.getchildren():
                    if (span.get("class").find("txt") != -1):
                        label = span.text

                # If we did find a label we have to look 
                # for a value 
                if (label and len(label) > 0):
                    next_td = all_tds[td_index + 1]

                    if (next_td.get("class").find("data") != -1):
                        # We have a data
                        for span in next_td.getchildren():
                            if (span.get("class").find("txt") != -1):
                                if (span.text):
                                    data = span.text
                                else:
                                    # If it is a link
                                    span_children = span.getchildren()
                                    if (span_children and len(span_children) > 0):
                                        data = span_children[0].text

                                # Include into dict
                                all_data[label] = data

                                # Erase it
                                label = ""
                                data = ""

    return all_data

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from tqdm.notebook import tqdm
#from selenium.webdriver.chrome.options import Options
import time



import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
def coletar_scrap():
    URL = 'https://statusinvest.com.br/acoes/busca-avancada'
    #output = 'busca-avancada.csv'

    #if path.exists(output):
    #    os.remove(output)

    #chrome_options = Options()
    #chrome_options.binary_location = GOOGLE_CHROME_BIN
    #chrome_options.add_argument('--disable-gpu')
    #chrome_options.add_argument('--no-sandbox')
    #driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)

    #driver = webdriver.Chrome('chromedriver/chromedriver.exe')

    #chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument('--headless')
    #chrome_options.add_argument('--no-sandbox')
    #chrome_options.add_argument('--disable-dev-shm-usage')
    #driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)

    gChromeOptions = webdriver.ChromeOptions()
    gChromeOptions.add_argument("window-size=1920x1480")
    gChromeOptions.add_argument("disable-dev-shm-usage")
    driver = webdriver.Chrome(
        chrome_options=gChromeOptions, executable_path=ChromeDriverManager().install()
    )
    
    #driver = webdriver.Chrome(ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install())
    #driver.get(URL)
    driver.get(URL)
    sleep(5)

    button_buscar = driver.find_element_by_xpath('//div/button[contains(@class,"find")]')

    button_buscar.click()
    sleep(5)

    button_skip = driver.find_element_by_xpath('//div/button[contains(@class,"btn-close")]')

    button_skip.click()
    sleep(5)

    button_download = driver.find_element_by_xpath('//div/a[contains(@class, "btn-download")]')
    button_download.click()
    sleep(1)
    
    #if path.exists(output):
               

    df = pd.read_csv('busca-avancada.csv', sep=';', decimal=',', thousands='.')
    driver.close()
    return df


import requests
def scrap_fundamentus():
    url  = 'http://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    result = requests.get(url, headers=headers)
    df = pd.read_html(result.content)[0]
    
    return df


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


# ----------------------------------BACKGROUND -------------------------------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------------------------SIDEBAR -------------------------------------------------------------
def main():

    set_background('bkgold.png')

    st.sidebar.header("Explorador de ativos")
    n_sprites = st.sidebar.radio(
        "Escolha uma opção", options=["Análise técnica e fundamentalista", "Comparação de ativos","Descobrir novos ativos", "Rastreador de trade", "Análise de carteira e previsão de lucro"], index=0
    )

    st.sidebar.markdown('É preciso ter paciência e disciplina para se manter firme em suas convicções quando o mercado insiste que você está errado.!')
    st.sidebar.markdown('Benjamin Graham')
    st.sidebar.markdown('Email para contato: lucas.vasconcelos3@gmail.com')
    #st.sidebar.markdown('Portfólio: https://github.com/lucasvascrocha')
    
# ------------------------------ INÍCIO ANÁLISE TÉCNICA E FUNDAMENTALISTA ----------------------------             

    if n_sprites == "Análise técnica e fundamentalista":

        #url  = 'http://www.fundamentus.com.br/resultado.php'
        #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        #result = requests.get(url, headers=headers)
        #df = pd.read_html(result.content)[0]
        #df = coletar_scrap()

        #print(df)
        #st.dataframe(df)




        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            st.image('https://media.giphy.com/media/rM0wxzvwsv5g4/giphy.gif', width=400)

        with col3:
            st.write("")
        
        #st.image('https://media.giphy.com/media/rM0wxzvwsv5g4/giphy.gif', width=400)    
        #image = Image.open('imagens/logo.jpg')
        #st.image(image, use_column_width=True)                       
        st.title('Análise Técnica e fundamentalista')
        st.subheader('Escolha o ativo que deseja analisar e pressione enter')
        nome_do_ativo = st.text_input('Nome do ativo')


        st.write('Este explorador funciona melhor para ações, porém também suporta alguns fundos imobiliários')    
        st.write('Os parâmetros utilizados em grande maioria foram seguindo as teorias de Benjamin Graham')

        if nome_do_ativo != "":
            nome_do_ativo = str(nome_do_ativo + '.SA')
            st.subheader('Analisando os dados')
            df = Ticker(nome_do_ativo,country='Brazil')
            time = df.history( period='max')
            #st.dataframe(time.tail())

# ------------------------------ RESUMO ---------------------------- 

            resumo = pd.DataFrame(df.summary_detail)
            resumo = resumo.transpose()
            if len(nome_do_ativo) == 8:
              fundamentus = get_specific_data(nome_do_ativo[:5])
              fundamentus = pd.DataFrame([fundamentus])
              
              try:

                    
                  pfizer = yf.Ticker(nome_do_ativo)
                  info = pfizer.info 
                  st.title('PERFIL DA EMPRESA')
                  st.subheader(info['longName']) 
                  st.markdown('** Setor **: ' + info['sector'])
                  st.markdown('** Atividade **: ' + info['industry'])
                  st.markdown('** Website **: ' + info['website'])


                  st.markdown('** P/L **: ' + fundamentus['P/L'][0])
                  st.markdown('** P/VP **: ' + fundamentus['P/VP'][0])
                  st.markdown('** Próximo pagamento de dividendo: **: ' + (pfizer.calendar.transpose()['Earnings Date'].dt.strftime('%d/%m/%Y')[0]))

                  st.markdown('** Recomendação: **: ' + info['recommendationKey'] )

                  
                  #st.markdown('** Alvo mínimo: **: ' + info['targetLowPrice'] )
                  #st.markdown('** Alvo máximo: **: ' + info['targetHighPrice'] )
                  #, '** Alvo mínimo: **: ' + info['targetLowPrice'], '** Alvo médio: **: ' + info['targetMeanPrice'], '** Alvo máximo: **: ' + info['targetHighPrice'])


                  with st.expander("Resumo atividade"):
                      st.markdown('** Resumo atividade **: ' + info['longBusinessSummary'])                  

                  #st.markdown('** Dividend Yield (%) -12 meses **: ' + round(info['dividendYield']*100,2))
            #  except:
            #    exit
                
            #  try:
            #      fundInfo = {
            #      'Dividend Yield (%) -12 meses': round(info['dividendYield']*100,2),
            #      'P/L': fundamentus['P/L'][0],
            #     'P/VP': fundamentus['P/VP'][0],
            #     'Próximo pagamento de dividendo:': (pfizer.calendar.transpose()['Earnings Date'].dt.strftime('%d/%m/%Y')[0]) }
            
            #    st.markdown('** P/L **: ' + fundamentus['P/L'][0])
            #      fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
            #      fundDF = fundDF.rename(columns={0: 'Valores'})
            #      st.subheader('Informações fundamentalistas') 
            #      st.table(fundDF)
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

              fig.update_layout(height=800, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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

              fig.update_layout(height=800, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig)
            
# ------------------------------ GRÁFICOS DE Retorno acumulado---------------------------- 

            layout = go.Layout(title="Retorno acumulado",xaxis=dict(title="Data"), yaxis=dict(title="Retorno"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Scatter(x=time.reset_index()['date'][-365:], y=time.reset_index()['close'][-365:].pct_change().cumsum(), mode='lines', line_width=3,line_color='rgb(0,0,0)'))
            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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

            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ Previsões---------------------------- 

            st.subheader('Previsões')

            st.write('As previsões são feitas levando em conta apenas o movimento gráfico, porém o movimento do preço de um ativo é influenciado por diversos outros fatores, com isso, deve se considerar as previsões como uma hipótese de o preço do ativo variar somente pela sua variação gráfica')

            st.write('Previsão considerando os últimos 365 dias, pode ser entendida como uma tendência dos dados segundo o último ano')
            
            st.write('Opção de alterar a previsão: caso esteja buscando resultados a curto prazo é possível alterar o "periodo analisado" para fazer previsões apenas com base nos últimos x dias. Neste caso o movimento gráfico para trás dos dias selecionados não serão levados em conta')
            periodo_analisado = int(st.number_input(label='período analisado (dias de resultados passados)',value=360))

            st.write('Opção de alterar a previsão: possibilidade de prever resultados futuros por mais de 30 dias')
            periodo_futuro = int(st.number_input(label='período futuro a prever (dias)',value=30))

            time = time.reset_index()
            time = time[['date','close']]
            time.columns = ['ds','y']

            #Modelling
            m = Prophet()
            #m.fit(time[-360:])
            m.fit(time[-periodo_analisado:])
            #future = m.make_future_dataframe(periods=30, freq='B')
            future = m.make_future_dataframe(periods= periodo_futuro, freq='B')
            #forecast = m.predict(future[-30:])
            forecast = m.predict(future[-periodo_futuro:])

            from fbprophet.plot import plot_plotly, plot_components_plotly

            fig1 = plot_plotly(m, forecast)
            fig1.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1)

            st.subheader('Tendência diária e semanal')
            st.write('0 = segunda, 1 = terça, ... , 5 = sábado, 6 = domingo')
            fig2 = m.plot_components(forecast,uncertainty = False,weekly_start=1)
            
            st.plotly_chart(fig2)
            

            #st.write('Previsão considerando as últimas semanas, pode ser entendida como uma tendência dos dados segundo os últimos dias. Leva em consideração diversos fatores como: Índice de força relativa RSI, oscilador estocástico %K, Indicador Willian %R além do movimento gráfico dos últimos dias')

            #predict = stocker.predict.tomorrow(nome_do_ativo)

            #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação feche no valor de: R$',f'{predict[0]}')

            #preço_ontem= round(time['y'][-1:].values[0],2)
            #if predict[0] < preço_ontem:
                #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação caia de ',f'{preço_ontem}', 'para valor de: R$ ',f'{predict[0]}')
            #else:
                #st.write('Previsão para o dia:',f'{predict[2]}','é que a ação suba de ',f'{preço_ontem}', 'para valor de: R$ ',f'{predict[0]}')
                         
# ------------------------------ INÍCIO Comparação de ativos ------------------------------------------------------------------------------------

    if n_sprites == "Comparação de ativos":
        
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            st.image('https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif', width=300)

        with col3:
            st.write("")

        #st.image('https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif', width=300)    
        #image = Image.open('imagens/logo.jpg')
        #st.image(image, use_column_width=True)                       
        st.title('Comparação de ativos')
        st.subheader('Escolha até 4 ativos para comparar')
        nome_do_ativo1 = st.text_input('Nome do 1º ativo')
        nome_do_ativo2 = st.text_input('Nome do 2º ativo')
        nome_do_ativo3 = st.text_input('Nome do 3º ativo')
        nome_do_ativo4 = st.text_input('Nome do 4º ativo')
        
        if nome_do_ativo4 != "":
            st.subheader('Analisando os dados')
            nome_do_ativo1 = str(nome_do_ativo1 + '.SA')
            nome_do_ativo2 = str(nome_do_ativo2 + '.SA')
            nome_do_ativo3 = str(nome_do_ativo3 + '.SA')
            nome_do_ativo4 = str(nome_do_ativo4 + '.SA')
            
            df = Ticker([nome_do_ativo1,nome_do_ativo2,nome_do_ativo3,nome_do_ativo4],country='Brazil')
            time = df.history( start='2018-01-01', end = (dt.datetime.today() + dt.timedelta(days=1)).strftime(format='20%y-%m-%d'))
            lista = get_data()
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

            comparar = todos.loc[todos.index.isin([nome_do_ativo1[:5],nome_do_ativo2[:5],nome_do_ativo3[:5],nome_do_ativo4[:5]])]
            
            st.dataframe(comparar)

# ------------------------------ INÍCIO Comparação DY ---------------
            
            layout = go.Layout(title="DY",xaxis=dict(title="Ativo"), yaxis=dict(title="DY %"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Bar(x=comparar.sort_values('DY',ascending=True).index, y=comparar.sort_values('DY',ascending=True)['DY'] ))

            fig.update_layout(autosize=False,width=800,height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação P/L ---------------

            layout = go.Layout(title="P/L",xaxis=dict(title="Ativo"), yaxis=dict(title="P/L"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Bar(x=comparar.sort_values('P/L',ascending=True).index, y=comparar.sort_values('P/L',ascending=True)['P/L'] ))

            fig.update_layout(autosize=False,width=800,height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação P/V---------------

            layout = go.Layout(title="P/VP",xaxis=dict(title="Ativo"), yaxis=dict(title="P/VP"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Bar(x=comparar.sort_values('P/VP',ascending=True).index, y=comparar.sort_values('P/VP',ascending=True)['P/VP'] ))

            fig.update_layout(autosize=False,width=800,height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação P/L * P/VP---------------

            layout = go.Layout(title="P/L X P/VP",xaxis=dict(title="Ativo"), yaxis=dict(title="P/L X P/VP"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Bar(x=comparar.index, y=comparar['P/L'] * comparar['P/VP'] ))

            fig.update_layout(autosize=False,width=800,height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE retorno acumulado---------------------------- 

            periodo_inicio = int(st.number_input(label='periodo retorno acumulado',value=360))

            ret = time.reset_index()
            layout = go.Layout(title="Retorno acumulado",xaxis=dict(title="Data"), yaxis=dict(title="Retorno"))
            fig = go.Figure(layout = layout)
            for i in range(len(ret['symbol'].unique())):
              fig.add_trace(go.Scatter(x=ret.loc[ret['symbol']==ret['symbol'].unique()[i]][-periodo_inicio:]['date'], y=ret.loc[ret['symbol']==ret['symbol'].unique()[i]][-periodo_inicio:]['close'].pct_change().cumsum(),mode='lines',name=ret.reset_index()['symbol'].unique()[i]))


            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE MÉDIAS MÓVEIS 50---------------------------- 

            rolling_50  = time['close'].rolling(window=50)
            rolling_mean_50 = rolling_50.mean()
            rolling_mean_50 = pd.DataFrame(rolling_mean_50.reset_index())
            # mm50 = time.reset_index()


            layout = go.Layout(title="MÉDIAS MÓVEIS 50",xaxis=dict(title="Data"), yaxis=dict(title="Preço R$"))
            fig = go.Figure(layout = layout)
            for i in range(len(rolling_mean_50['symbol'].unique())):
              fig.add_trace(go.Scatter(x=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['date'], y=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['close'],mode='lines',name=time.reset_index()['symbol'].unique()[i]))


            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE MÉDIAS MÓVEIS 20---------------------------- 

            rolling_50  = time['close'].rolling(window=20)
            rolling_mean_50 = rolling_50.mean()
            rolling_mean_50 = pd.DataFrame(rolling_mean_50.reset_index())
            # mm50 = time.reset_index()


            layout = go.Layout(title="MÉDIAS MÓVEIS 20",xaxis=dict(title="Data"), yaxis=dict(title="Preço R$"))
            fig = go.Figure(layout = layout)
            for i in range(len(rolling_mean_50['symbol'].unique())):
              fig.add_trace(go.Scatter(x=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['date'], y=rolling_mean_50.loc[rolling_mean_50['symbol']==rolling_mean_50['symbol'].unique()[i]]['close'],mode='lines',name=time.reset_index()['symbol'].unique()[i]))


            fig.update_layout(autosize=False,width=800,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE volatilidade--------------------------- 

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

            fig.update_layout(autosize=False,width=800,height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE sharpe_ratio--------------------------- 

            sharpe_ratio = returns.mean()/volatility
            sharpe = pd.DataFrame(sharpe_ratio).reset_index()
            sharpe = sharpe.dropna()

            layout = go.Layout(title=f"SHARP (Risco / Volatilidade)",xaxis=dict(title="Data"), yaxis=dict(title="Sharp"))
            fig = go.Figure(layout = layout)
            for i in range(len(sharpe['symbol'].unique())):
              fig.add_trace(go.Scatter(x=sharpe.loc[sharpe['symbol']==sharpe['symbol'].unique()[i]]['date'], y=sharpe.loc[sharpe['symbol']==sharpe['symbol'].unique()[i]]['close'],name=sharpe['symbol'].unique()[i] ))

            fig.update_layout(autosize=False,width=800,height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE correlação-------------------------- 
            st.subheader('Correlação')
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
            fig = px.imshow(retscomp.corr(),color_continuous_scale='YlOrRd')
            fig.update_layout(autosize=False,width=1200,height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

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
            fig.update_layout(autosize=False,width=800,height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação de ativos ------------------------------------------------------------------------------------

    if n_sprites == "Descobrir novos ativos":
        
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            st.image('https://media.giphy.com/media/3ohs4gux2zjc7f361O/giphy.gif', width=400)

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

        lista = get_data()
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
            
            
            
            
# ------------------------------ INÍCIO Rastreador de trade ------------------------------------------------------------------------------------

    if n_sprites == "Rastreador de trade":
        
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            st.image('https://media.giphy.com/media/d83YIjgW4uyTpYfjbd/giphy.gif', width=400)

        with col3:
            st.write("")

        #st.image('https://media.giphy.com/media/d83YIjgW4uyTpYfjbd/giphy.gif', width=400)    
        #image = Image.open('imagens/logo.jpg')
        #st.image(image, use_column_width=True)  
        
        lista = get_data()
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
        
# ------------------------------ INÍCIO Análise de carteira ------------------------------------------------------------------------------------        
        


    if n_sprites == "Análise de carteira e previsão de lucro":
        
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.write("")

        with col2:
            st.image('https://media.giphy.com/media/3ShFD4IvX96027JjhH/giphy.gif', width=400)

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
                    image = Image.open('b3.png')
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

                    features, rank = rank(X, y)

                    st.subheader('Estas variáveis são as que mais impactam nas decisões da inteligência artificial')

                    st.table(rank['features'].head(10).reset_index(drop=True))


                    #fazendo previsão em toda a bolsa

                    lista = get_data()
                    todos = pd.DataFrame(flatten(lista).keys()).transpose()
                    todos.columns = todos.iloc[0]
                    for i in range(len(lista)):
                        todos = pd.concat([todos,pd.DataFrame(lista[i]).transpose()])

                    todos = todos.iloc[1:]
                    todos['name'] = (todos.index + '.SA' )

                    data = yf.download(list(todos['name']),start=date_year_ago)

                    df_filled = pd.DataFrame(columns = ['name'])
                    df_filled['name'] = todos['name']

                    def inputer_data(data, df_filled):
                        lista = list(df_filled.index)
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



                    df_filled = inputer_data(data, df_filled)
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

            







            
            
            #st.subheader('Faça upload aqui do seu extrato da B3')
            #file  = st.file_uploader('Entre com seu extrato (.xlsx)', type = 'xlsx')    
            #if file:
                #df = pd.read_excel(file)

                #st.dataframe(df)
                #st.table(df)
                #st.write('Lucro Total até hoje: R$',round(df['Preço Médio (Venda)'].sum(),2))
                #extrato = pd.DataFrame(df)

                #nome_do_ativo = str(df['Código de Negociação'][0] + '.SA')
                #st.subheader('Analisando os dados')
                #df = Ticker(nome_do_ativo,country='Brazil')
                #time = df.history( period='max')
                #time = df.history( start= extrato['Período (Inicial)'][0])
                #st.dataframe(time.tail())
                #st.write(datetime.strptime(extrato['Período (Inicial)'][0], '%d/%m/%Y').strftime('%Y-%m-%d'))
                #datetimeobject = datetime.strptime(extrato['Período (Inicial)'],'%d/%m/%Y')
                #newformat = datetimeobject.strftime('%Y-%m-%d')
                #st.write(newformat)


                        





# ------------------------------ FIM ----------------------------

        
if __name__ == '__main__':
    main()

