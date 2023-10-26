import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas_datareader.data as pdr
yf.pdr_override()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.metrics import r2_score
from statistics import NormalDist as nd

import pytz
import datetime as dt
from datetime import date, datetime
import matplotlib.dates as mdates
my_year_month_fmt = mdates.DateFormatter('%d/%m/%y')

class Trend:
    def __init__(self, tickers, pos = None, benchmark = None, inicio = dt.datetime.today().date()-dt.timedelta(365), fim = dt.datetime.today().date()):
        ''' 
        Definir pode definir o inicio da série histórica, 
        tickers a serem levados em consideração (incluíndo o benchmark que será automaticamente renomeado)
        Assim como o fim da série e as posição para cada ativos
        '''
        self.inicio = inicio 
        self.fim = fim
        self.tickers = tickers
        self.benchmark = benchmark

        if pos == None:
            self.pos = {i:'C' for i in tickers}
        else:
            self.pos = pos

        self.Data = pdr.get_data_yahoo(self.tickers, start=self.inicio, end=self.fim)['Adj Close'].bfill()

        if type(self.Data) == type(pd.Series([], dtype='object')):
            self.Data = pd.DataFrame(self.Data.values, index=self.Data.index,columns=self.tickers)
        if self.benchmark != None:
            if len(self.benchmark) > 1:
                for i in self.benchmark:
                    self.Data[f'Benchmark {i}'] = pdr.get_data_yahoo(i, start=self.inicio, end=self.fim)['Adj Close'].bfill()
                    
            else:
                self.Data['Benchmark'] = pdr.get_data_yahoo(self.benchmark, start=self.inicio, end=self.fim)['Adj Close'].bfill()

        self.Data = self.Data.bfill().fillna(0)

    def retornos(self, tipo = 'pct', dist = ['Long', 'Short']):
        '''
        Talvez o mais complicado de entender seja o retorno EWM
        Ele se baseia em uma média móvel exponencial
        Dando mais peso aos dados mais recentes
        
        Semelhante, porém diferente, do Weighted normal
        Cria-se pesos de [1, 2, ..., n] em que n é o
        número de dados na série histórica.
        Normaliza-se e se aplica os pesos aos retornos
        '''
        self.retorno = self.Data.pct_change().fillna(0)
        lista = []
        f=0
        if type(self.retorno) == type(pd.Series([], dtype='object')):
            if self.pos[list(self.pos.keys())[0]].capitalize() == 'V':
                self.retorno = self.retorno * -1
        else:
            for i in self.retorno.columns:
                if i in list(self.pos.keys()):
                    if self.pos[i].capitalize() == 'V':
                        self.retorno[i] = self.retorno[i] * -1

        if tipo.capitalize()[0] == 'P': # Return
            return self.retorno
    
        elif tipo.capitalize()[0] == 'A': # Acumulated
            return ((1+self.retorno).cumprod())
    
        elif tipo.capitalize()[0] == 'W': # Weighted Return
            self.retorno['Pesos'] = [1+i for i in range(len(self.retorno))]
            self.retorno['Pesos'] = self.retorno['Pesos']/self.retorno['Pesos'].iloc[-1]
            self.retorno = self.retorno.apply(lambda x: x*self.retorno['Pesos'])
            self.retorno = self.retorno.drop('Pesos', axis=1)
            return self.retorno

        # Exponential Weights Return
        elif tipo.capitalize()[0] == 'E': 
            if type(dist) == type([]) and len(dist) == len([" ", " "]): #standard
                for i in dist:
                    # If string, must be Long. If Num, must bem int and the max of the list
                    if type(i) == type(str):
                        if str(i).capitalize()[0] == 'L':
                            self.ret_s = self.retorno.ewm(alpha=0.15, min_periods=132, adjust=False).mean()
                        elif str(i).capitalize()[0] == 'S':
                            self.ret_l = self.retorno.ewm(alpha=0.7, min_periods=22, adjust=False).mean()
                    elif type(i) == type(int()):
                        if i == max(dist):
                            self.ret_l = self.retorno.ewm(alpha=0.7, min_periods=i, adjust=False).mean()
                        elif i == min(dist):
                            self.ret_s = self.retorno.ewm(alpha=0.15, min_periods=i, adjust=False).mean()     
                    else:
                        print(f'ERROR {i} {type(i)}')
                # self.retorno = pd.concat([self.ret_s , self.ret_l], keys=['Long', 'Short'], names=['Tipo'], axis = 1) 
                # return self.retorno
                return self.ret_l, self.ret_s

    def formatar(self, valor):
        # Copiado do Edu
        return "{:,.2%}".format(valor)

    def di(self, JGP = False, Date = None):
        '''
        Essa função puxa o DI
        Ou simula a Risk Free Rate da competição JGP 2023
        '''
        self.Date = Date
        if self.Date == None:
            self.Date = pd.to_datetime(self.Data.index).strftime('%d/%m/%y')
        if JGP:
            self.Download = pd.DataFrame(columns = ['DI'])
            self.Download['DI'] = [np.array((.05 * (1/360))-1) for i in range((pd.to_datetime(self.Data.index[-1])-pd.to_datetime(self.Data.index[0])).days)]
        else:
            DI = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=csv&dataInicial={self.Date[0]}&dataFinal={self.Date[-1]}'
            self.Download = pd.read_csv(DI)
            self.Download.columns = ['DI']
            self.Download.index = self.Download.index.str.rstrip(f';"0')
            self.Download['DI'] = self.Download['DI'].str.rstrip(f'"').astype(int)/100000000
        return self.Download

    def medio(self, dias=5, ret = None):
        '''
        Retorna uma média móvel
        Nesse padrão, está totalmente configurado ao EWM
        O foco desse projeto
        '''
        if type(dias) == type({}):
            dias = list(dias.values())
        if ret == None:
            self.long, self.short = self.retornos('E', dist=dias)
        return self.long, self.short
        
    def mediana(self, dias=5, ret = None):
        '''
        Retorna uma mediana móvel
        Nesse padrão, está totalmente configurado ao EWM
        O foco desse projeto
        '''
        if type(dias) == type({}):
            dias = list(dias.values())
        if ret == None:
            ret = self.retornos('E', dist=dias)
        return ret.rolling(dias).median()
        
    def trend(self, dias = {'Long':126, 'Short':22}):
        '''
        Aqui, se calcula os padrões, a partir das médias
        móveis 'Long' e 'Short' que se referem, por padrão,
        a 126 e 22 dias, respectivamente.
        A subtração entre a Short e a Long leva, na teoria,
        a discrepância do mercado em relação a essas duas métricas.
        
        Seguindo a base teórica do projeto, aqui se torna possível
        a localização de tendências 'Buy' e 'Sell'
        '''
        med_l, med_s = self.medio(dias=dias)
        self.dif = (med_s-med_l).dropna()
        return self.dif, med_l, med_s,  self.retornos('A')
    
    def ordens(self):
        '''
        Aqui, efetivamente concluímos as condições para 'Buy', 'Sell'
        e 'Sem mudança', este último se refere a inconclusão dessa análise
        de qualquer efeito do deslocamento da curva.
        '''
        tr = self.trend()
        x = tr[0]
        ordens = pd.DataFrame(columns = x.columns, index = tr[0].index)
        for i in ordens.columns:
            ordens.loc[((x[i]>0) & (x[i].shift(1)<0)),i] = 'Buy'
            ordens.loc[((x[i]<0) & (x[i].shift(1)>0)),i] = 'Sell'
            ordens.loc[(~((x[i]<0) ^ (x[i].shift(1)<0))),i] = 'Sem mudança'
        return ordens.fillna('Sem mudança')
    
    def var(self, tipo = 'Param', dias = {'Long':126, 'Short':22}, confianca = 95):
        '''
        Criamos aqui o VaR paramétrico com os retornos EWM.
        Serão fundamental na base de construção dos Stop Gain e Stop Loss.
        '''
        self.ret_l, self.ret_s = self.retornos('E', dist=list(dias.values()))
        if type(tipo) != type([]):
            tipo = [tipo]
        if type(confianca) != type([]):
            confianca = [confianca]
        self.var = pd.DataFrame(index = self.ret_l.columns.values)
        for t in tipo:
            t = t.capitalize()[0]
        for i in confianca:
            if t == 'H':
                self.var[f'VaR Hist {i}%'] = {a:np.percentile(self.ret_l[a], (100-i)) for a in self.ret_l.columns}
            elif t == 'P':
                self.var[f'VaR Param {i}%'] = {a:self.ret_l[a].mean() - self.ret_l[a].std()*nd(0, 1).inv_cdf(i/100) for a in self.ret_l.columns}
            else:
                self.var['ERROR'] = []
        return self.var
    
    def test(self):
        '''
        Aqui emularemos o fundo em ação.
        Teremos as cotações de quando o fundo entrou na posição.
        Caso o contrário, entraremos em DI.
        '''
        Dt = self.Data
        ordem = self.ordens()
        carteira = pd.DataFrame(0, columns = ordem.columns, index=ordem.index)
        for i in carteira.columns:
            carteira.loc[((ordem[i] == 'Buy')), i] = Dt.loc[((ordem[i]=='Buy')), i]
        
        # for i in ordens.index:
        #     for _ in ordens.loc[i].values:
        #         if tr[0].loc[i] > 0:
                
        # return ordens
        
    
    # Refazer Trend do Cid: | Check
    # Fazer Weighted VaR | Check
    # Fazer Weighted | Check
    
    # Entrar a partir de certo coeficiente angular (a partir dos retornos da livre de risco de cada região dos ativos) |
    # Fazer "estabilidade" do C.A. dos retornos (5 dias atras) para 5 dias seguidos e pegar o desvio padrão e validar ou não stop gain |

    #Pegar X23 por cada dia e comparar com cotacao dol BRl (cap. derivativos Ana Clara)
    
Lista = ['EWG', 'GLD', 'TLT', 'SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XTN', 'EWJ', 'EEM', 'EWZ', 'FXE']
Vendidos = [""]

PPP = {i:"" for i in Lista}
for i in Lista:
  if i in Vendidos:
    PPP[i] = 'V'
  else:
    PPP[i] = 'C'

x = Trend(Lista, PPP, ['^BVSP'], inicio='2003-01-01').test()

print(x)

# y = ((1+y).cumprod())
# # px.area(x).show()
# fig, ax = plt.subplots(1,1,figsize=(12,6))

# for i in z.columns:
#     ax.set_title(i)
#     ax.plot(y['Long'][i])
#     # ax.plot(y['Short'][i])
#     ax.plot(z[i])
#     ax.set_xlim(y['Long'][i].index[0], y['Long'][i].index[-1])
#     # ax.plot(z)
#     plt.show()

# x, y = Trend(Lista, PPP, ['^BVSP']).medio(dias = {'Long':126, 'Short':22})

# print(x)
# print(y)