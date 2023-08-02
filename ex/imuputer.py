# ref: SimpleImputer
# 【scikit-learn】SimpleImputerで欠損値を補完し、統計情報を保存する
# https://zerofromlight.com/blogs/detail/70/


import os,sys
import numpy as np
import pandas as pd
from pathlib import Path
cwd = Path(__file__).parent.parent
out_path = cwd / "out"
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

import random
random.seed(0)

class PlotGraph:
    def __init__(self,save_name= os.path.basename(__file__).split(".")[0]):
        
        self.f,self.ax = plt.subplots(figsize=(12,7))
        self.out_path = os.path.join(str(out_path) ,f"{save_name}.png")
        
    def save(self):
        self.f.savefig(self.out_path,bbox_inches="tight")
        plt.close()


def generate_data():
    df = pd.DataFrame(np.arange(1, 26).reshape(5, 5), columns=list('ABCDE'))
    
    null_values = random.sample(list(np.arange(1, 26)),7)
    df = df.replace(null_values,np.nan)
    # print(list(np.arange(1, 26)))
    print(df)
    return df

def main():
    df = generate_data()
    pg = PlotGraph()
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df,columns =list('ABCDE'))
    for c in df.columns:
        pg.ax.plot(df[c],label=c)
    pg.ax.legend()
    pg.save()
    

if __name__ == "__main__":
    main()

        
        

