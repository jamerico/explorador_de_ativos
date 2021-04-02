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



# ------------------------------ SCRAPPING -------------------------------


def get_data(*args, **kwargs):
    url = 'http://www.fundamentus.com.br/resultado.php'
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201'),
                         ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')]

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
    url = "http://www.fundamentus.com.br/detalhes.php?papel=" + stock
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201'),
                         ('Accept', 'text/html, text/plain, text/css, text/sgml, */*;q=0.01')]
    
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

# ----------------------------------SIDEBAR -------------------------------------------------------------
def main():

    st.sidebar.header("Explorador de ativos")
    n_sprites = st.sidebar.radio(
        "Escolha uma opção", options=["Análise técnica e fundamentalista", "Comparação de ativos","Descobrir novos ativos"], index=0
    )

    st.sidebar.markdown('É preciso ter paciência e disciplina para se manter firme em suas convicções quando o mercado insiste que você está errado.!')
    st.sidebar.markdown('Benjamin Graham')
    st.sidebar.markdown('Email para contato: lucas.vasconcelos3@gmail.com')
    st.sidebar.markdown('Portfólio: https://github.com/lucasvascrocha')                                    

# ------------------------------ INÍCIO ANÁLISE TÉCNICA E FUNDAMENTALISTA ----------------------------             

    if n_sprites == "Análise técnica e fundamentalista":
        st.image('https://media.giphy.com/media/rM0wxzvwsv5g4/giphy.gif', width=400)    
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
            st.dataframe(time.tail())

# ------------------------------ RESUMO ---------------------------- 

            resumo = pd.DataFrame(df.summary_detail)
            resumo = resumo.transpose()
            if len(nome_do_ativo) == 8:
              fundamentus = get_specific_data(nome_do_ativo[:5])
              fundamentus = pd.DataFrame([fundamentus])
              
              pfizer = yf.Ticker(nome_do_ativo)
              info = pfizer.info 
              st.title('PERFIL DA EMPRESA')
              st.subheader(info['longName']) 
              st.markdown('** Setor **: ' + info['sector'])
              st.markdown('** Atividade **: ' + info['industry'])
              st.markdown('** Website **: ' + info['website'])
              
              try:
                fundInfo = {
            'Dividend Yield (%) -12 meses': round(info['dividendYield']*100,2),
            'P/L': fundamentus['P/L'][0],
            'P/VP': fundamentus['P/VP'][0],
            'Próximo pagamento de dividendo:': (pfizer.calendar.transpose()['Earnings Date'].dt.strftime('%d/%m/%Y')[0])
        }   
                fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
                fundDF = fundDF.rename(columns={0: 'Valores'})
                st.subheader('Informações fundamentalistas') 
                st.table(fundDF)
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

              fig.update_layout(height=800, showlegend=False)

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

              fig.update_layout(height=800, showlegend=False)

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
            fig.update_layout(autosize=False,width=800,height=800,)
            st.plotly_chart(fig)
            
# ------------------------------ GRÁFICOS DE Retorno acumulado---------------------------- 

            layout = go.Layout(title="Retorno acumulado",xaxis=dict(title="Data"), yaxis=dict(title="Retorno"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Scatter(x=time.reset_index()['date'][-365:], y=time.reset_index()['close'][-365:].pct_change().cumsum(), mode='lines', line_width=3,line_color='rgb(0,0,0)'))
            fig.update_layout(autosize=False,width=800,height=800,)

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE Médias móveis---------------------------- 

            rolling_50  = time['close'].rolling(window=50)
            rolling_mean_50 = rolling_50.mean()

            rolling_20  = time['close'].rolling(window=20)
            rolling_mean_20 = rolling_20.mean()

            rolling_10  = time['close'].rolling(window=10)
            rolling_mean_10 = rolling_10.mean()

            layout = go.Layout(title="Médias móveis",xaxis=dict(title="Data"), yaxis=dict(title="Preço R$"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=time["close"][-120:], mode='lines', line_width=3,name='Real',line_color='rgb(0,0,0)'))
            fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=rolling_mean_50[-120:],mode='lines',name='MM(50)',opacity = 0.6))
            fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=rolling_mean_20[-120:],mode='lines',name='MM(20)',opacity = 0.6))
            fig.add_trace(go.Scatter(x=time.reset_index()['date'][-120:], y=rolling_mean_10[-120:],mode='lines',name='MM(10)',opacity = 0.6,line_color='rgb(100,149,237)'))
            # fig.add_trace(go.Candlestick(x=time.reset_index()['date'][-120:], open=time['open'][-120:],high=time['high'][-120:],low=time['low'][-120:],close=time['close'][-120:]))
            fig.update_layout(autosize=False,width=800,height=800,)

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE Retração de Fibonacci---------------------------- 

            time_fibo = time.copy()
            periodo_fibonacci = int(st.number_input(label='periodo fibonacci',value=90))
            
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
            fig.update_layout(autosize=False,width=800,height=800,)

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

            fig.update_layout(autosize=False,width=800,height=800,)

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE pivôs---------------------------- 

            periodo_pivo = int(st.number_input(label='periodo RSI',value=20))

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
            fig.update_layout(autosize=False,width=800,height=800,)

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
            fig.update_layout(autosize=False,width=800,height=800,)

            st.plotly_chart(fig)

# ------------------------------ Previsões---------------------------- 

            st.subheader('Previsões')

            st.write('As previsões são feitas levando em conta apenas o movimento gráfico, porém o movimento do preço de um ativo é influenciado por diversos outros fatores, com isso, deve se considerar as previsões como uma hipótese de o preço do ativo variar somente pela sua variação gráfica')

            st.write('Previsão considerando os últimos 365 dias, pode ser entendida como uma tendência dos dados segundo o último ano')

            time = time.reset_index()
            time = time[['date','close']]
            time.columns = ['ds','y']

            #Modelling
            m = Prophet()
            m.fit(time[-360:])
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future[-30:])

            from fbprophet.plot import plot_plotly, plot_components_plotly

            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
            #st.plotly_chart(m, forecast)
            fig2 = m.plot_components(forecast)
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

        st.image('https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif', width=300)    
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

            fig.update_layout(autosize=False,width=800,height=400,)

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação P/L ---------------

            layout = go.Layout(title="P/L",xaxis=dict(title="Ativo"), yaxis=dict(title="P/L"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Bar(x=comparar.sort_values('P/L',ascending=True).index, y=comparar.sort_values('P/L',ascending=True)['P/L'] ))

            fig.update_layout(autosize=False,width=800,height=400,)

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação P/V---------------

            layout = go.Layout(title="P/VP",xaxis=dict(title="Ativo"), yaxis=dict(title="P/VP"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Bar(x=comparar.sort_values('P/VP',ascending=True).index, y=comparar.sort_values('P/VP',ascending=True)['P/VP'] ))

            fig.update_layout(autosize=False,width=800,height=400,)

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação P/L * P/VP---------------

            layout = go.Layout(title="P/L X P/VP",xaxis=dict(title="Ativo"), yaxis=dict(title="P/L X P/VP"))
            fig = go.Figure(layout = layout)
            fig.add_trace(go.Bar(x=comparar.index, y=comparar['P/L'] * comparar['P/VP'] ))

            fig.update_layout(autosize=False,width=800,height=400,)

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE retorno acumulado---------------------------- 

            periodo_inicio = int(st.number_input(label='periodo retorno acumulado',value=360))

            ret = time.reset_index()
            layout = go.Layout(title="Retorno acumulado",xaxis=dict(title="Data"), yaxis=dict(title="Retorno"))
            fig = go.Figure(layout = layout)
            for i in range(len(ret['symbol'].unique())):
              fig.add_trace(go.Scatter(x=ret.loc[ret['symbol']==ret['symbol'].unique()[i]][-periodo_inicio:]['date'], y=ret.loc[ret['symbol']==ret['symbol'].unique()[i]][-periodo_inicio:]['close'].pct_change().cumsum(),mode='lines',name=ret.reset_index()['symbol'].unique()[i]))


            fig.update_layout(autosize=False,width=800,height=800,)

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


            fig.update_layout(autosize=False,width=800,height=800,)

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


            fig.update_layout(autosize=False,width=800,height=800,)

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

            fig.update_layout(autosize=False,width=800,height=400,)

            st.plotly_chart(fig)

# ------------------------------ GRÁFICOS DE sharpe_ratio--------------------------- 

            sharpe_ratio = returns.mean()/volatility
            sharpe = pd.DataFrame(sharpe_ratio).reset_index()
            sharpe = sharpe.dropna()

            layout = go.Layout(title=f"SHARP (Risco / Volatilidade)",xaxis=dict(title="Data"), yaxis=dict(title="Sharp"))
            fig = go.Figure(layout = layout)
            for i in range(len(sharpe['symbol'].unique())):
              fig.add_trace(go.Scatter(x=sharpe.loc[sharpe['symbol']==sharpe['symbol'].unique()[i]]['date'], y=sharpe.loc[sharpe['symbol']==sharpe['symbol'].unique()[i]]['close'],name=sharpe['symbol'].unique()[i] ))

            fig.update_layout(autosize=False,width=800,height=400,)

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

            plt.figure(figsize=(10,8))
            sns.heatmap(retscomp.corr(),annot=True)

            st.pyplot()

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
            fig.update_layout(autosize=False,width=800,height=600,)

            st.plotly_chart(fig)

# ------------------------------ INÍCIO Comparação de ativos ------------------------------------------------------------------------------------

    if n_sprites == "Descobrir novos ativos":

        st.image('https://media.giphy.com/media/3ohs4gux2zjc7f361O/giphy.gif', width=400)    
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


# ------------------------------ FIM ----------------------------

        
if __name__ == '__main__':
    main()

