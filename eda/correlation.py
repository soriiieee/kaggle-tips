
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

import matplotlib.pyplot as plt


"""
2023.7.24 kaggle で役に立ったもの
2023.8.2 更新
"""

class Correlation:
    def __init__(self,output_folder=None):
        if output_folder is None:
            self.output_folder = Path(__file__).parent
        else:
            self.output_folder = output_folder
        print("start")
        print("output = ",self.output_folder)
        
    @classmethod
    def get_corr_with_columns(cls,df,corr_cols,target_col="Class",isABS=False):
        """_summary_
        https://www.kaggle.com/code/datafan07/icr-simple-eda-baseline/input
        
        Args:
            df (_type_): DataFrame
            corr_cols (_type_): 説明変数を持った一覧(連続である必要あり)
            target_col (_type_): 目的変数
        """
        
        correlations = df.loc[:,corr_cols].corrwith(df[target_col]).to_frame()
        
        if isABS:
            print("ABS mode!!")
            correlations['Corr'] = correlations[0].abs()
        else:
            correlations['Corr'] = correlations[0]
            
        sorted_correlations = correlations.sort_values('Corr', ascending=False)['Corr']
        
        return sorted_correlations

    @classmethod
    def plot_corr_with_columns(cls,df,corr_cols,target_col="Class",isABS=False,threshold=0.1):
        sorted_correlations = cls.get_corr_with_columns(df,corr_cols,target_col="Class",isABS=isABS)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(sorted_correlations.iloc[1:].to_frame()[sorted_correlations>=threshold], cmap='inferno', annot=True, vmin=-1, vmax=1, ax=ax)
        
        print("use-columns")
        df_use = ((sorted_correlations>=threshold)>0).astype(int).reset_index()
        cols =df_use[df_use["Corr"]==1]["index"].values.tolist()
        print(cols)
        plt.title('Feature Correlations With Target')
        plt.show()
    
    
    def all_corr_with_columns(self,df,corr_cols,target_col="Class",isABS=False)
    ### memo 
        """
        .unstack()について
        	AB	AF	AH	AM	AR	AX	AY	AZ	BC	BD	...	FI	FL	FR	FS	GB	GE	GF	GH	GI	GL
            AB	1.000000	0.350231	0.249246	0.530687	0.157712	0.471912	0.011004	0.158569	0.331736	0.278920	...	0.004967	0.169934	0.017990	0.036913	0.318741	0.003327	0.017192	0.148456	0.002903	0.061601
            AF	0.350231	1.000000	0.044140	0.183961	0.044917	0.248439	0.039417	0.219172	0.058831	0.045430	...	0.021914	0.148289	0.021009	0.003108	0.079993	0.076449	0.159796	0.187105	0.140935	0.127870
            AH	0.249246	0.044140	1.000000	0.128268	0.7
        のようなDataFrmaeに対して、以下のように、一列で格納することができる
        AB  AB    1.000000
            AF    0.350231
            AH    0.249246
            AM    0.530687
            AR    0.157712
                    ...   
        GL  AB    0.061601
            AF    0.127870
            AH    0.029175  
        
        
        """
        correlations = df.loc[:,corr_cols].corr().abs().unstack().sort_values(kind="quicksort",ascending=False).reset_index()
        correlations = correlations[correlations['level_0'] != correlations['level_1']] #preventing 1.0 corr
        corr_max=correlations.level_0.head(150).tolist()
        corr_max=list(set(corr_max)) #removing duplicates

        corr_min=correlations.level_0.tail(34).tolist()
        corr_min=list(set(corr_min)) #removing duplicates


        correlation_train = train.loc[:,corr_max].corr()
        mask = np.triu(correlation_train.corr())

        plt.figure(figsize=(30, 12))
        sns.heatmap(correlation_train,
                    mask=mask,
                    annot=True,
                    fmt='.3f',
                    cmap='coolwarm',
                    linewidths=0.00,
                    cbar=True)


        plt.suptitle('Features with Highest Correlations',  weight='bold')
        plt.tight_layout()
        plt.close()
        
    def corr_futures(self,df,num_cols):
        corr = df.loc[:, num_cols].corr()
        sns.clustermap(corr, metric="correlation", cmap="hot", figsize=(20, 20))
        plt.suptitle('Correlations Between Features', fontsize=24, weight='bold')
        plt.show()
        
        
        
        