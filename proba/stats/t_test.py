"""
223.8.9
ref : https://www.learning-nao.com/?p=2620

母集団の分散＝標準偏差などは、知らないことが多い-z値が使えないので
この場合は、標本の普遍分散をとり、（小さくなる）t値、正規分布よりもやや大きめ
をとって、信頼区間とする

それぞれの実験で得られたデータから計算された興味あるパラメーターが、
その信頼区間に収まっている実験の頻度が、95回である

T検定
２標本の違いがあるかどうかの検定？
１：母集団が正規分布に従う
２：母集団の分散が同じ（２標本で）
"""

import pandas as pd
from sklearn.datasets import load_iris

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as ss

def figset(numgfig=1):
    
    if numgfig==1:
        f,ax = plt.subplots(figsize=(12,8))
    else:
        f,ax = plt.subplots(1,numgfig,figsize=(12,8))
    png = "./sample.png"
    return f,ax,png

iris=load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["class"] = iris.target
# print(df.head())

p0 = df[df["class"]==0]["petal length (cm)"]
p1 = df[df["class"]==1]["petal length (cm)"]

# f,ax,png = figset()
# sns.kdeplot(data = df,x = "petal length (cm)" , hue="class",ax=ax)
# f.savefig(png,bbox_inches="tight")

#前提確認
def check_assumption(p0,p1):
    """_summary_
    Args:
        p0 (_DataSeries_): _description_
        p1 (_DataSeries_): _description_
    """
    ## データの概要把握
    print("Size : ",len(p0),"--", len(p1))
    print("MEANS: ",round(p0.mean(),4),"--", round(p1.mean(),4))
    print("MEANS: ",round(p0.std(),4), "--", round(p1.std(),4))
    
    ## データの正規分布の確認(Q-Q plot)
    # f,ax,png = figset(2)
    # ss.probplot(p0, plot=ax[0])
    # ss.probplot(p1, plot=ax[1])
    # f.savefig(png,bbox_inches="tight")
    
    ##shapiro-wirk検定
    _,p_0 = ss.shapiro(p0)
    _,p_1 = ss.shapiro(p1)
    print("Sharipo-test: ",round(p_0,4), "--", round(p_1,4))
    
    ## Welch -t-test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
    result = ss.ttest_ind(p0,p1,equal_var = False)
    print(result) #t検定量/p-value
    
    

check_assumption(p0,p1)




