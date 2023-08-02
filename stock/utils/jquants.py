
import os
import sys

import requests
import json

from utils.utils import write_json_file,json_read

from pathlib import Path
import pandas as pd

from pprint import pprint

cwd = Path(__file__).parent.parent
# def write_json_file(target, output_dir):
#     with open(output_dir, 'w') as f:
#         json.dump(target, f)


class JQuants:
    def __init__(self,config) -> None:
        self.config = config
        self.get_token()
        
        self.list_jp = "jp_stock.json"
        self.list_jp_csv = "jp_stock.csv"
        
        
        self.nikkei225 = cwd / "tbl/nikkei225.csv"
        

    def get_token(self):
        if self.config["api"]["idToken"] is None:
            try:
                data={"mailaddress":self.config["user"]["email"], "password": self.config["user"]["password"]}
                r_post = requests.post("https://api.jquants.com/v1/token/auth_user", data=json.dumps(data))
                self.refresh_token = r_post.json()['refreshToken']
                r_post = requests.post(f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={self.refresh_token}")
                
                self.idToken = r_post.json()["idToken"]
                print(self.idToken)
                
            except Exception as e:
                self.idToken = None
                print(f"Not Get refresh token...{e}")
                exit(1)
        else:
            print("already get idToken...")
            self.idToken = self.config["api"]["idToken"]
            print(self.idToken)
        
    def get_jp_list(self):
        if self.idToken is None:
            self.get_token()
        headers = {'Authorization': 'Bearer {}'.format(self.idToken)}
        r = requests.get("https://api.jquants.com/v1/listed/info", headers=headers)
        if r.status_code == 200:
            data = r.json()
            # write_json_file(r.json(),self.list_jp)
            df = json_read(data)
            df.to_csv(self.list_jp_csv,index=False)
    
    def json_to_csv_save(self):
        df = json_read(self.list_jp)
        
    
    def search_code(self,query):
        
        if not os.path.exists(self.nikkei225):
            print(self.nikkei225)
            exit(1)
        
        df = pd.read_csv(self.nikkei225)
        df = df[df["CompanyName"].str.contains(query)]
        
        if df.shape[0]==0:
            print("Not Companies...")
            exit(1)
        else:
            df = df[["Code","CompanyName","Sector17CodeName","Sector33CodeName"]].set_index("Code")
            results = {}
            for code,v in df.iterrows():
                results[code] = list(v)
            pprint(results)

    
    def req_financial_data(self,code,info):
        headers = {'Authorization': 'Bearer {}'.format(self.idToken)}
        r = requests.get("https://api.jquants.com/v1/fins/statements?code={}".format(str(code)+ "0"), headers=headers)
        
        if r.status_code == 200:
            data = r.json()["statements"][0]
            
            TypeOfCurrentPeriod = data["TypeOfCurrentPeriod"]
            
            NetSales = data["NetSales"]                 #売上
            OperatingProfit = data["OperatingProfit"]   #営業利益
            OrdinaryProfit = data["OrdinaryProfit"]     #経常利益
            Profit = data["Profit"]                     #当期純利益
            EPS = data["EarningsPerShare"]              #一株あたり当期純利益
            print(info[0] , TypeOfCurrentPeriod , NetSales , OperatingProfit , OrdinaryProfit , Profit,EPS)
    
            
        
        
            
            
        
        