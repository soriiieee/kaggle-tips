"""
223.8.9
ref : ヴァイズ統計
chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.jstage.jst.go.jp/article/jasj/75/6/75_351/_pdf

信頼区間と確信区間の違い（区別）
https://qiita.com/katsu1110/items/4e8529f01a1389c03712


"""
import pandas as pd
from sklearn.datasets import load_iris

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as ss
from scipy.special import comb
import numpy as np

def figset(numgfig=1):
    
    if numgfig==1:
        f,ax = plt.subplots(figsize=(12,8))
    else:
        f,ax = plt.subplots(1,numgfig,figsize=(12,8))
    png = "./sample.png"
    return f,ax,png

def binomial_dist(N,n,p):
    L = comb(N,n) * (p**n)*(1-p)**(N-n)
    return L

## 尤度の確認
p = np.linspace(0,1,200)
Ls = binomial_dist(30,21,p)


nmin,nmax = ss.binom.interval(0.95,30,0.7)

print(nmin,nmax)
print(ss.binom.cdf(int(nmin),30,0.7) , "~",ss.binom.cdf(int(nmax),30,0.7) )
f,ax,png = figset(1)
ax.plot(p,Ls,"o")
f.savefig(png,bbox_inches="tight")



#二項分布
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
# mean, var, skew, kurt = ss.binom.stats(30, 0.7, moments='mvsk')
# print(mean, var, skew, kurt)

# xs = np.arange(0,30,1)
# ps = ss.binom.pmf(xs,30,0.7)



# f,ax,png = figset(1)
# ax.plot(xs,ps,"o")
# f.savefig(png,bbox_inches="tight")

