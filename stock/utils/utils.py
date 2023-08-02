import requests
import json
import pandas as pd

def write_json_file(obj, output_json):
    with open(output_json, 'w') as f:
        json.dump(obj, f,ensure_ascii=False,indent=4)
        

def json_read(json_object):
    # with open(output_json, 'r') as f:
    #     data = json.loads(f.read())
    # print(data["info"])
    coms = json_object["info"]
    
    res = {}
    for ii,com in enumerate(coms):
        ticker = com["Code"]
        res[ticker] = com
    
    df = pd.DataFrame(res).T
    return df


def make_categories():
    
    df = pd.read_csv("../tbl/jp_stock.csv")
    dict17 = df[['Sector17Code','Sector17CodeName']].drop_duplicates().sort_values("Sector17Code").\
        set_index("Sector17Code").to_dict()['Sector17CodeName']
    dict33 = df[['Sector33Code','Sector33CodeName']].drop_duplicates().sort_values("Sector33Code").\
        set_index("Sector33Code").to_dict()['Sector33CodeName']
    
    write_json_file(dict17,"../tbl/cate17.json")
    write_json_file(dict33,"../tbl/cate33.json")


def make_nikkei225_comps():
    df = pd.read_csv("../tbl/jp_stock.csv")
    df["Code"] = df["Code"].apply(lambda x: int(str(x)[:4]))
    
    # df = df[df["MarketCodeName"]=='プライム']
    # print(df["MarketCodeName"].unique())
    with open("../tbl/nikkei225.txt" ,"r") as f:
        codes = [ int(c.replace("\n","")) for c in f.readlines()]
    
    df = df.loc[df["Code"].isin(codes),:]
    df.to_csv("../tbl/nikkei225.csv",index=False)
    
    

    

if __name__ == "__main__":
    make_nikkei225_comps()
    
    
        
        
    