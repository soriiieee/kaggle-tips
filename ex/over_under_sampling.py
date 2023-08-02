# ref: SimpleImputer　23.8.1
# 機械学習における不均衡データへの対処方法としてアンダーサンプリングやオーバーサンプリング
# https://book-read-yoshi.hatenablog.com/entry/2021/07/31/imbalanced_data_smote#google_vignette

import os,sys
import numpy as np
import pandas as pd
from pathlib import Path
cwd = Path(__file__).parent.parent
out_path = cwd / "out"

import pickle
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Dense, Activation, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import plot_model #(model, "my_first_model_with_shape_info.png", show_shapes=True)
    
except Exception as e:
    print(e)
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE

import warnings
warnings.simplefilter("ignore")

import random

def reset_random_seeds(SEED=123):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
reset_random_seeds()

class PlotGraph:
    def __init__(self,save_name= os.path.basename(__file__).split(".")[0]):
        
        self.f,self.ax = plt.subplots(figsize=(12,7))
        self.out_path = os.path.join(str(out_path) ,f"{save_name}.png")
        
    def save(self):
        self.f.savefig(self.out_path,bbox_inches="tight")
        plt.close()

class SamplerShuffler:
    def __init__(self,target_name="Class"):
        self.target_name = target_name
        
    def standard_and_to_numpy(self,x_train,y_train,x_valid=None,y_valid=None):
        # 平均、標準偏差計算
        X_train_mean = x_train.mean()
        X_train_std  = x_train.std()
        # データの標準化
        x_train = (x_train - X_train_mean) / X_train_std
        x_train,y_train  = x_train.values, y_train.values
        
        # valid - data
        if x_valid is not None:
            x_valid  = ( x_valid - X_train_mean)/ X_train_std
            x_valid,y_valid = x_valid.values,y_valid.values
        else:
            x_valid,y_valid = None,None
        return x_train, y_train, x_valid, y_valid

    def shuffle_in_unison(self , a, b):
        """
        # 学習用データをシャッフルするための関数 - シャッフルしないと、model.fitのvalidation_split時に目的変数に偏りが発生してしまうため
        Args:
            a (numpy.ndarry): X
            b (numpy.ndarry): y

        Returns:
            X,y: shuffled
        """
        assert len(a) == len(b)
        shuffled_a  = np.empty(a.shape, dtype=a.dtype)
        shuffled_b  = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def sample_data(self,x_train,y_train,mode="over"):
        class_num_dict = y_train.reset_index()[self.target_name].value_counts().to_dict() #{0: 284315, 1: 492}
        if mode == "under":
            count_min = min(class_num_dict[0] , class_num_dict[1])
            sampling_strategy = {0: np.round(count_min * 9).astype(int), 1: count_min } # class0:class1 = 9:1 * 数の設定も行う
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)
            x_train, y_train = sampler.fit_resample(x_train, y_train)
            return x_train,y_train

        elif mode == "over":
            count_max = max(class_num_dict[0] , class_num_dict[1])
            sampler = RandomOverSampler(sampling_strategy={ 0: count_max,  1: count_max //9 }, random_state=0)
            x_train, y_train = sampler.fit_resample(x_train, y_train)
            return x_train, y_train
        
        elif mode == "SMOTE":
            count_max = max(class_num_dict[0] , class_num_dict[1])
            sampler = SMOTE(sampling_strategy={ 0: count_max,  1: count_max //9 },
                            k_neighbors=5, random_state=0)
            x_train, y_train = sampler.fit_resample(x_train, y_train)
            return x_train, y_train

class Trainer:
    
    def __init__(self):
        self.epoch = 100
        self.batch_size = 126
        self.callbacks = EarlyStopping(monitor='val_loss', patience=3, mode='auto')
    
    def get_model(self,X,y):
        model = Sequential()
        model.add(InputLayer(input_shape=X[1].shape))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model
    
    def fit(self,x_train,y_train,x_valid=None,y_valid=None,save_pkl=None):
        
        self.model = self.get_model(x_train,y_train)
        # モデルのコンパイル
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=self.epoch,
                       batch_size=self.batch_size,
                       validation_split=0.3,
                       verbose=2, callbacks=self.callbacks
                       )
        
        if save_pkl:
            self.save_model(save_pkl)
        return
    
    def save_model(self,save_pkl):
        with open(save_pkl , "wb") as f:
            pickle.dump(self.model,f)
        print(save_pkl)
    
    def load_model(self,save_pkl):
        with open(save_pkl , "rb") as f:
            model = pickle.load(f)
        return model
        


def generate_data(sampler="under"):
    
    df = pd.read_csv(os.path.join(str(cwd) , "data" , "creditcard.csv"))
    
    # print(df[["V1","V2" , "Amount" , "Class"]].head())
    X,y = df.drop(["Class"],axis=1) ,df["Class"]
    
    class_num_dict = y.reset_index()["Class"].value_counts().to_dict() #{0: 284315, 1: 492}
    print("train-data size: ",class_num_dict) #train-data size:  {0: 199020, 1: 344}
    return X,y


class Metrics:
    def __init__(self):
        pass
    
    # 混同行列を見やすくする関数。以下を参考とした。
    # https://qiita.com/makaishi2/items/9fb6bf94daa6208c8ed0
    @staticmethod
    def cm(y_true,pred,columns=["0","1"]):
        
        matrix = confusion_matrix(y_true, pred)
        # '正解データ'をn回繰り返すリスト生成
        act = ['T'] * len(columns)
        pred = ['P'] * len(columns)

        #データフレーム生成
        cm = pd.DataFrame(matrix, columns=[pred, columns], index=[act, columns])
        return cm
    
    # テストデータの予測精度を表示
    @staticmethod
    def show(y_true,pred):
        print("accuracy_score:{:.5f}".format(accuracy_score(y_true, pred)))
        print("precision_score:{:.5f}".format(precision_score(y_true, pred)))
        print("recall_score:{:.5f}".format(recall_score(y_true, pred)))
        print("f1_score:{:.5f}".format(f1_score(y_true, pred)))


def main():
    
    X,y = generate_data()
    x_train,x_valid,y_train,y_valid = train_test_split(X,y,stratify=y,test_size=0.3)
    SS = SamplerShuffler(target_name="Class")
    
    # print("train-data = ",x_train.shape,"valid-data = ",x_valid.shape)
    x_train,y_train = SS.sample_data(x_train,y_train,mode="over") #sample-data
    # print(x_train.shape,y_train.value_counts())
    x_train,y_train,x_valid,y_valid = SS.standard_and_to_numpy(x_train,y_train,x_valid,y_valid)
    x_train,y_train = SS.shuffle_in_unison(x_train,y_train)
    print(x_train.shape,x_valid.shape)
    
    
    
    tr = Trainer()
    # tr.fit(x_train,y_train,save_pkl = "../data/model.pkl")
    model = tr.load_model("../data/model.pkl")
    y_pred = model.predict(x_valid)
    y_pred = np.round(y_pred)
    
    matrix = Metrics.cm(y_valid,y_pred)
    
    print(matrix)
    Metrics.show(y_valid,y_pred)
    #----
    pg = PlotGraph()


if __name__ == "__main__":
    main()

        
        

