import numpy as np
import pandas as pd

def describe_df(df):
    ndata = df.shape[0]
    for c in df.columns:
        print("-"*100)
        p= round(df[c].nunique()*100 / ndata,1)
        print("colname:({}) dtypes: ({}) nunique: ({})[{}%]".format(
            c,df[c].dtypes ,df[c].nunique(),p))
        if df[c].nunique()<15:
            print(df[c].unique())
            print(df[c].value_counts() / ndata)
    return 


def detail_df(df,unique_num=10):
    """ 2023.7.31 update for kaggle challenge"""
    for ii,c in enumerate(df.columns):
        
        print("-"*5)
        if df[c].dtypes == object:
            if df[c].nunique() > unique_num:
                text = "column [{}]: {}({}) data(nunique={})ex: {}"\
                    .format(ii,c,df[c].dtypes,df[c].nunique(),list(df[c].sample(3).values))
            else:
                text = "column [{}]: {}({}) class(nunique={}): {}".format(ii,c,df[c].dtypes,df[c].nunique(),list(df[c].unique()))
                                                              
        elif df[c].dtypes == float or df[c].dtypes == int:
            text = "column [{}]: {}({}) data(ex): {}\n".format(ii,c,df[c].dtypes,list(df[c].sample(5).values))
            text+= "            min={}|mean={}|max={}    - std={}"\
            .format(
                round(df[c].min(),3),
                round(df[c].mean(),3),
                round(df[c].max(),3),
                round(df[c].std(),3),
            )
        else:
            text = "column [{}]: {}({}) data(ex): {}".format(ii,c,df[c].dtypes,list(df[c].sample(5).values))
        
        print(text)
    return 