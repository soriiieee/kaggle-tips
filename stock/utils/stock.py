"""_summary_
2023.7.16 ysorimachi
Get Stock data from yfinance API
"""

import os
import yfinance as yf

# https://pandas-datareader.readthedocs.io/en/latest/readers/nasdaq-trader.html?highlight=tickers#pandas_datareader.nasdaq_trader.get_nasdaq_symbols
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols #
import pandas_datareader as web

from datetime import datetime,date ,timedelta
# how to use -> https://pypi.org/project/yfinance/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16


class StockData:
    def __init__(self,ticker="AAPL",out_folder =None):
        
        
        
        
        try:
            int(ticker[:4])
            self.nation = "JP"
            self.ticker = ticker
        except Exception:
            self.nation = "US"
            self.ticker = ticker
            self.obj = yf.Ticker(self.ticker)
        print("ticker = " , self.ticker)
        # print("info = " , self.obj.info)
        self.out_folder = out_folder
        
        self.price_csv = os.path.join(self.out_folder,"price",f"{self.ticker}.csv")
        self.out_png = os.path.join(self.out_folder,"png",f"{self.ticker}.png")
        
    def get_daily(self,period=365):
        print(f"過去{period}日前のデータから抽出します")
        ## period = days
        end = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=period)).strftime("%Y-%m-%d")
            
        if self.nation == "US":
            data = self.obj.history(start = start,end=end).reset_index()
            data["Date"] = data["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
            return data[["Date","Close"]].set_index("Date")
        elif self.nation == "JP":
            data = web.DataReader("{}.JP".format(self.ticker), data_source='stooq', start=start, end=end).reset_index()
            data["Date"] = data["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
            return data[["Date","Close"]].set_index("Date")
    
    def ratio_data_from_start(self,period=365,method="ratio"):
        data = self.get_daily(period)
        
        if method == "ratio":
            return data["Close"]/data["Close"].values[0]
        elif method == "diff":
            return data["Close"].diff().dropna()
        elif method == "logdiff": #対数収益率
            return np.log(data["Close"]).diff().dropna()
    
    ##plot functions
    def plot(self,data):
        f,ax = plt.subplots(figsize=(15,8))
        if isinstance(data.index[0], str):
            data.index = [datetime.strptime(t, "%Y-%m-%d").date() for t in data.index]
        ax.plot(data,marker = "o" , label = self.ticker)
        ax.legend()
        f.savefig(self.png_output, bbox_inches="tight")
        return None
    
    # @classmethod
    # def listTickers(cls):
    #     tickers = get_nasdaq_symbols(retry_count=3, timeout=30, pause=None)
    #     return tickers


if __name__ == "__main__":
    tickers = get_nasdaq_symbols(retry_count=3, timeout=40, pause=None)
    print(tickers)
    

        
        
    