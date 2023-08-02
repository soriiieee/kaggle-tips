import os
import sys

import configparser

from utils.stock import StockData
from utils.jquants import JQuants
from watch import WatchList


if __name__ == "__main__":
    
    config_parameter = configparser.ConfigParser()
    config_parameter.read("config.ini",encoding='utf-8')
    
    jq = JQuants(config_parameter)
    
    ## 一度作成したら使わない ---
    # jq.get_jp_list()
    
    ## 文字列で検索する ---
    # jq.search_code("三菱")
    
    ## データ検索 ---
    datas = WatchList.data
    for code , info in datas.items():
        print(code,info)
        # --- 財務情報の取得 ---
        # jq.req_financial_data(code,info)
        
        # --- 株価情報の取得 ---
        st = StockData(str(code),out_folder = "./data")
        
        df = st.get_daily(365*10)
        df.to_csv(st.price_csv)
    
        


